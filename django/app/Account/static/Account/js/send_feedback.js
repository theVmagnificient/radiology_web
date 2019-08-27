function sendFeedbackForm() {
    const url = '/home/feedback';
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

    feedback_form = '<div class="row">' +
        '<form class="col s12">' +
        '<div class="input-field col s12">' +
        '<i class="material-icons prefix">title</i>' +
        '<input id="feedback_title" name="title" type="text" class="validate">' +
        '<label for="feedback_title">Заголовок</label>' +
        '</div>' +
        '<div class="input-field col s12">' +
        '<i class="material-icons prefix">message</i>' +
        '<textarea id="feedback_text" name = "text" class="materialize-textarea"></textarea>' +
        '<label for="feedback_text">Сообщение</label>' +
        '</div>' +
        '</div>';

    Swal.fire({
        title: 'Отправить письмо',
        html: feedback_form,
        showCancelButton: true,
        confirmButtonText: 'Отправить',
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                let title = document.getElementById("feedback_title").value
                let text = document.getElementById("feedback_text").value

                if (title.length == 0 || text.length == 0) {
                    reject("Заполните все поля!");
                }
                $.ajax({
                    type: "POST",
                    url: url,
                    data: {
                        title: title,
                        text: text,
                    },
                    success: (function (response) {
                        if (!response["success"]) {
                            reject("Похоже на сервере возникла ошибка");
                        } else {
                            resolve("Письмо отправлено");
                        }
                    }),
                });
            }).then(value => {
                console.log(value)
            }, reason => {
                Swal.showValidationMessage(
                    `Ошибка: ${reason}`
                )
            })
        },
        allowOutsideClick: () => !Swal.isLoading()
    }).then((result) => {
        if (result.value) {
            Swal.fire({
                type: 'success',
                title: 'Письмо успешно отправлено!',
                showConfirmButton: false,
                timer: 1500
            })
        }
    })
}


$(document).keypress(function (event) {
    let sendFormButton = document.getElementById("predictFormButton")
    var keycode = (event.keyCode ? event.keyCode : event.which);
    if (keycode == '13' && sendFormButton != null) {
        sendFormButton.click()
    }
});