const el = document.querySelector(".pageWrapper");

var movementStrength = 15;
var height = movementStrength / $(window).height()
var width = movementStrength / $(window).width()

el.addEventListener("mousemove", (e) => {
	var pageX = e.pageX - ($(window).width() / 2);
    var pageY = e.pageY - ($(window).height() / 2);
    var newvalueX = width * pageX * -1 - 25;
    var newvalueY = height * pageY * -1 - 50;
  	el.style.setProperty('--x', newvalueX  + "px");
  	el.style.setProperty('--y', newvalueY + "px");
});

function sendForm() {
	let re = /^[a-zA-Z0-9]+$/;

	response = document.getElementById("response")
	login = document.getElementById("login").value
	password = document.getElementById("password").value

	if (login.length == 0 || password.length == 0) {
		response.innerHTML = "<p class = 'error_msg'>Заполните все поля!</p>";
	} else if(!re.test(login)) {
		response.innerHTML = "<p class = 'error_msg'>Такого логина не существует!</p>"
	} else if(!re.test(password)) {
		response.innerHTML = "<p class = 'error_msg'>Неправильный пароль!</p>"
	} else {
		response.innerHTML = "<p class = 'success_msg'>А здесь происходит AJAX запрос</p>";
	}
}