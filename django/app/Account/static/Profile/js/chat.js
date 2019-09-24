let loc = window.location;

let userName = loc.pathname.split("/")[2];
let endpoint = (loc.protocol == "https:" ? "ws://" : "ws://") + loc.host + "/chat/" + userName + "/";
let socket = new ReconnectingWebSocket(endpoint);
let myLogin = $("input#myLogin").val();
let myID = $("input#myID").val();

class Messenger {
  constructor() {
    this.messageList = [];
    this.deletedList = [];

    this.me = 1; // completely arbitrary id
    this.them = 5; // and another one

    this.onRecieve = message => console.log('Recieved: ' + message.text);
    this.onSend = message => console.log('Sent: ' + message.text);
    this.onDelete = message => console.log('Deleted: ' + message.text);
  }

  send(text = '') {
    text = this.filter(text);

    if (this.validate(text)) {
      let message = {
        user: this.me,
        text: text,
        time: new Date().getTime() };


      this.messageList.push(message);

      this.onSend(message);
    }
  }

  recieve(text = '') {
    text = this.filter(text);

    if (this.validate(text)) {
      let message = {
        user: this.them,
        text: text,
        time: new Date().getTime() };


      this.messageList.push(message);

      this.onRecieve(message);
    }
  }

  delete(index) {
    index = index || this.messageLength - 1;

    let deleted = this.messageLength.pop();

    this.deletedList.push(deleted);
    this.onDelete(deleted);
  }

  filter(input) {
    let output = input.replace('bad input', 'good output'); // such amazing filter there right?
    return output;
  }

  validate(input) {
    return !!input.length; // an amazing example of validation I swear.
  }}


class BuildHTML {
  constructor() {
    this.messageWrapper = 'message-wrapper';
    this.circleWrapper = 'circle-wrapper';
    this.textWrapper = 'text-wrapper';

    this.meClass = 'me';
    this.themClass = 'them';
  }

  _build(text, who) {
    return `<div class="${this.messageWrapper} ${this[who + 'Class']}">
              <div class="${this.circleWrapper} animated bounceIn"></div>
              <div class="${this.textWrapper}">...</div>
            </div>`;
  }

  me(text) {
    return this._build(text, 'me');
  }

  them(text) {
    return this._build(text, 'them');
  }}


$(document).ready(function () {
  let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")
  let messenger = new Messenger();
  let buildHTML = new BuildHTML();

  let $input = $('#input');
  let $send = $('#send');
  let $content = $('#content');
  let $inner = $('#inner');

  function safeText(text) {
    $content.find('.message-wrapper').last().find('.text-wrapper').text(text);
  }

  function animateText() {
      $content.find('.message-wrapper').last().find('.text-wrapper').addClass('animated fadeIn');

    // setTimeout(() => {
    //   $content.find('.message-wrapper').last().find('.text-wrapper').addClass('animated fadeIn');
    // }, 350);
  }

  function scrollBottom() {
    $($inner).animate({
      scrollTop: $($content).offset().top + $($content).outerHeight(true) },
    {
      queue: false,
      duration: 'ease' });

  }

  function buildSent(message) {
    $content.append(buildHTML.me(message.text));
    safeText(message.text);
    animateText();

    scrollBottom();
  }

  function buildRecieved(message) {
    $content.append(buildHTML.them(message.text));
    safeText(message.text);
    animateText();

    scrollBottom();
  }

  function sendMessage() {
    let text = $input.val();

    let socketData = {
      "message": text,
    }
    socket.send(JSON.stringify(socketData));

    // messenger.send(text);

    $input.val('');
    $input.focus();
  }

  messenger.onSend = buildSent;
  messenger.onRecieve = buildRecieved;

  $input.focus();

  // $send.on('click', function (e) {
  //   sendMessage();
  // });

  socket.onmessage = function(e) {
    let chatData = JSON.parse(e.data);
    console.log(chatData, typeof chatData)

    processMsg = function(fields) {
      if (fields["sender"] == myID) { 
        messenger.send(`${fields["message"]}`)
      } else {
        messenger.recieve(`${fields["message"]}`);
      }
    }

    if (Array.isArray(chatData)) {
      chatData.forEach(function (msg, index) {
        processMsg(msg["fields"]);
      });
    } else {
      processMsg(chatData["fields"]);
    }
  }
  socket.onopen = function(e) {
    console.log("open", e);

    $input.on('keydown', function (e) {
      let key = e.which || e.keyCode;

      if (key === 13) {// enter key
        e.preventDefault();

        sendMessage();
      }
    });
  }
  socket.onerror = function(e) {
    console.log("error", e);
  }
  socket.onclose = function(e) {
    console.log("close", e);
  }
});

