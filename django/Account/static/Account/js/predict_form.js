function sendPredictForm() {
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    const url = '/home/predict/get';

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
    swal({
        title: "Выполнить предсказание",
        text: "Это может занять некоторое время",
        icon: "info",
        showCancelButton: true,
        closeOnConfirm: false,
        showLoaderOnConfirm: true,
        button: {
            text: "Отправить",
            closeModal: false,
        },
    }).then(function () {
        $.ajax({
            type: "POST",
            url: url,
            data: {
                temp_data: "temp_data",
            },
            success: (function(response) {
                if (!response["success"]) {
                    swal("Ошибка!", "Похоже на сервере возникла ошибка", "error");
                } else {
                    swal("Выполнено", "Предсказания посчитаны. " + response["message"], "success");
                }
            }),
        });
    });
}


$(document).keypress(function (event) {
    let sendFormButton = document.getElementById("predictFormButton")
    sendFormButton.press
    var keycode = (event.keyCode ? event.keyCode : event.which);
    if (keycode == '13') {
        sendFormButton.click()
    }
});
