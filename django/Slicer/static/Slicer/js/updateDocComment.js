function sendDocComment() {
  const url = "/series/changeDocComment";
  comment = $("#doc_comment").val().replace(/(\r\n|\n|\r)/gm,"");
  let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

  function csrfSafeMethod(method) {
      // these HTTP methods do not require CSRF protection
      return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
  }
  $.ajaxSetup({
      beforeSend: function (xhr, settings) {
          if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
              xhr.setRequestHeader("X-CSRFToken", token);
          }
      }
  });
  $.ajax({
      type: "POST",
      url: url,
      data: {
          id: SERIES_ID,
          comment: comment,
      },
      success: (function (response) {
          console.log(response);
          if (!response["ok"]) {
            M.toast({html: response["msg"]});
          } else {
            M.toast({html: "Комментарий сохранён"});
          }
      }),
  });
} 