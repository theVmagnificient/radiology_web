function changeAccountInfo() {
    const url = '/home/changeAccount';
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

    let changeAccountInfoForm = '<div class="row">' +
        '<form class="col s12">' +
            '<div class="input-field col s12">' +
                '<input id="new_name" name="new_name" type="text" class="validate" value="'+ ACCOUNT_NAME +'">' +
                '<span class="helper-text">Ваше имя</span>' + 
            '</div>' +
            '<div class="input-field col s12">' +
                '<input id="new_surname" name="new_surname" type="text" class="validate" value="'+ ACCOUNT_SURNAME +'">' +
                '<span class="helper-text">Ваша фамилия</span>' + 
            '</div>' +
            '<div class="input-field col s12">' +
                '<input id="new_login" name="new_login" type="text" class="validate" value="'+ ACCOUNT_LOGIN +'">' +
                '<span class="helper-text">Ваш логин</span>' + 
            '</div>' +
            '<div class="input-field col s12">' +
            '<input id="new_mail" name="new_mail" type="text" class="validate" value="'+ ACCOUNT_MAIL +'">' +
            '<span class="helper-text">Ваша почта</span>' + 
            '</div>' +
        '</form>' +
    '</div>';

    Swal.fire({
        title: 'Измменить данные об аккаунте',
        html: changeAccountInfoForm,
        showCancelButton: true,
        confirmButtonText: 'Отправить',
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                let name = document.getElementById("new_name").value
                let surname = document.getElementById("new_surname").value
                let login = document.getElementById("new_login").value
                let mail = document.getElementById("new_mail").value

                let data = []
                if (name != ACCOUNT_NAME) {
                    data.push({"name": name});
                }
                if (surname != ACCOUNT_SURNAME) {
                    data.push({"surname": surname});
                }
                if (login != ACCOUNT_LOGIN) {
                    data.push({"login": login});
                }
                if (mail != ACCOUNT_MAIL) {
                    data.push({"mail": mail});
                }
                console.log(data);
                console.log("ASDSDDSD")

                if (name.length == 0 || surname.length == 0 || login.length == 0 || mail.length == 0) {
                    reject("Заполните все поля!");
                }
                $.ajax({
                    type: "POST",
                    url: url,
                    data: data,
                    success: (function (response) {
                        if (!response["success"]) {
                            reject("Похоже на сервере возникла ошибка");
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
                title: 'Данные успешно изменены!',
                showConfirmButton: false,
                timer: 1500
            })

            window.location = "/home"
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