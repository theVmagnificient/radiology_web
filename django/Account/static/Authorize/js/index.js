function sendForm() {

	let responseContainer = document.getElementById("response")
	let login = document.getElementById("login").value
	let password = document.getElementById("password").value
	let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

	const url = '/auth/login/';
	function csrfSafeMethod(method) {
		// these HTTP methods do not require CSRF protection
		return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
	}
	$.ajaxSetup({
		beforeSend: function(xhr, settings) {
			if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
				xhr.setRequestHeader("X-CSRFToken", token);
			}
		}
	});
	$.ajax({
		type: "POST",
		url: url,
		data: {
			login: login,
			passwd: password,
		},
		success: (function(response) {
			if (!response['success']) {
				showValidate(response['message']);
			}
			if (response['redirect']) {
				location.href = response['redirect'];
			}
		}),
	});
}