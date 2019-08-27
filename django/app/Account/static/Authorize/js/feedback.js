function send_feedback() {
	const url = "/feedback/";
	let name = $("#appointment_name").val();
	let email = $("#appointment_email").val();
	let text = $("#appointment_message").val();
	
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

    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value");
    
    $.ajax({
        type: "POST",
        url: url,
        data: {
            name: name,
            mail: email,
            text: text,
        },
        success: (function (response) {
            if (!response["ok"]) {
                Swal.fire({
                    type: 'error',
                    title: 'Ошибка на сервере',
                    showConfirmButton: false,
                })
            } else {
                Swal.fire({
                    type: 'success',
                    title: 'Письмо успешно отправлено!',
                    showConfirmButton: false,
                })
            }
        }),
        error: (function (response) {
        	console.log("ERROR: ", response);
        }),
    });
}