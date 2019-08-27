function uploadAva() {
    const url = '/home/uploadAva';
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    let uploadAvaForm = '<form id = "uploadAvaForm" action="/home/uploadAva" enctype="multipart/form-data">' +
        '<div class="file-field input-field">' +
            '<div class="btn">' +
                '<span>Загрузите файл</span>' +
                '<input type="file" name="file" id="file">' +
            '</div>' +
            '<div class="file-path-wrapper">' +
                '<input class="file-path validate" type="text" placeholder="Загрузить">' +
            '</div>' +
        '</div>' +
    '</form>';

    Swal.fire({
        title: 'Загрузить фотографию',
        html: uploadAvaForm,
        confirmButtonText: 'Отправить',
        showCancelButton: true,
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                let form = document.getElementById('uploadAvaForm');
                let formData = new FormData(form);

                if (file.value.length == 0) {
                    reject("Вы не прикрепили файл")
                }

                formData.append('file', file);
                
                let xhr = new XMLHttpRequest();
                xhr.open('POST', form.getAttribute('action'), true);
                xhr.setRequestHeader("X-CSRFToken", token);

                xhr.responseType = 'json';
                xhr.onload = function () {
                    if (xhr.readyState === xhr.DONE) {
                        if (xhr.status === 200) {
                            if (!xhr.response.ok) {
                                reject(xhr.response.msg)
                            } 
                        }
                    }
                };

                xhr.send(formData);
            }).then(value => {
                console.log(value)
            }, reason => {
                Swal.showValidationMessage (
                    `Ошибка: ${reason}`
                )
            })
        },
        allowOutsideClick: () => !Swal.isLoading()
    }).then((result) => {
        if (result.value) {
            Swal.fire({
                type: 'success',
                title: 'Файл был успешно загружен!',
                showConfirmButton: false,
                timer: 1500
            }).then(() => {
                window.location = "/home";
            });
        }
    })
}