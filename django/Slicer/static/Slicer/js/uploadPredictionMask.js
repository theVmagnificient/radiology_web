let loc = window.location;
let endpoint = (loc.protocol == "https:" ? "ws://" : "ws://") + loc.host + "/uploadPredictionMask/";
let socket;
let timeoutWebsocketReqID;
let uploadContainer = $("#maskUploadProgressContainer");

function startDelayedWebsocketRequest(socketData) {
    timeoutWebsocketReqID = window.setInterval(function() {
        socket.send(JSON.stringify(socketData));
    }, 1000);
}

function stopDelayedWebsocketRequest() {
    window.clearInterval(timeoutWebsocketReqID);
}

function showUploadButton() {
    $("#uploadButtonContainer").html(`
        <a class="waves-effect waves-light btn-small" onclick="uploadPredictionMask()">
            <i class="material-icons left">file_upload</i>
            Загрузить маску
        </a>
    `);
}

function hideUploadButton() {
    $("#uploadButtonContainer").html("");
}

function renderLoadingMask(status, progress=-1) {
    if (uploadContainer.html() == "") {
        let htmlContent = `
        <div id = "progressUploadingResearch">
            <p style="statusText" id = "statusText"><b>Статус:</b> ${status}</p>
        </div>
        <div class="progress">
        `
        if (progress == -1) {
            htmlContent += `<div id = "progressBar" class="indeterminate"></div>`;
        } else {
            htmlContent += `<div id = "progressBar" class="determinate" style="width: ${progress}%"></div>`;
        }
        htmlContent += `</div>`;

        uploadContainer.html(htmlContent);
    } else {
        $("#progressUploadingResearch").html(`<p><b>Статус: </b> ${status}</p>`);
        if (progress == -1) {
            $("#progressBar").removeClass("determinate");
            $("#progressBar").addClass("indeterminate");
        } else {
            $("#progressBar").removeClass("indeterminate");
            $("#progressBar").addClass("determinate");
            $("#progressBar").css("width", `${progress}%`)
        }
    }
}

function followMaskProcessProgress(response) {
    if (response["ok"]) {
        Swal.fire(
            'Файл загружен',
            'Файл отправлен на обработку. С этого шага можно безопасно уходить.',
            'success'
        );

        socket = new ReconnectingWebSocket(endpoint);
        socket.onopen = function(e) {
            renderLoadingMask("Соединение установлено. Получение данных...", -1);
            let socketData = {
                "maskID": response["maskID"],
            }
            startDelayedWebsocketRequest(socketData);
            hideUploadButton();
        }
        socket.onmessage = function(e) {
            let data = JSON.parse(e["data"]);
            console.log(data);

            let progress = parseFloat(data["progress"])
            let roundedProgress = Math.round(progress)
            let status = parseInt(data["status"])

            if (status == 1) {
                renderLoadingMask(`Извлекаем данные о исследовании... `, -1);
            } else if (status == 2) {
                renderLoadingMask(`Накладываем маску. ${roundedProgress}%`, progress);
            } else if (status == 3){
                renderLoadingMask(`Работаем с базой данных...`, -1);
            } else if (status == 5) {
                renderLoadingMask("Исследование обработано. Обновляем данные...", -1);
                stopDelayedWebsocketRequest()
                window.setTimeout(function() {
                    location.reload();
                }, 2000);
            }
            else {
                renderLoadingMask("На сервере возникла ошибка.", -1);
            }

        }
        socket.onerror = function(e) {
            renderLoadingMask("Возникла ошибка на сервере. Обработка не завершена.", 0);
            stopDelayedWebsocketRequest();
            showUploadButton();
        }
        socket.onclose = function(e) {
            renderLoadingMask("Соединение разорвано. Переподключаемся...", -1);
            stopDelayedWebsocketRequest();
        }
    } else {
        Swal.fire(
            "Ошибка",
            response["message"],
            "error",
        );
    }
}

function uploadPredictionMask() {
    const url = '/series/uploadPredictionMask';
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    let uploadAvaForm = `
    <form id = "uploadResearchForm" action="/series/uploadPredictionMask" enctype="multipart/form-data">'
        <div class="row">
            <div class="input-field col s12">
                <input id="mask_name" type="text" class="validate">
                <label for="mask_name">Название маски</label>
            </div>
        </div>
      <div class="row">
        <form class="col s12">
          <div class="row">
            <div class="input-field col s12">
              <textarea id="mask_description" class="materialize-textarea"></textarea>
              <label for="mask_description">Комментарий к маске</label>
            </div>
          </div>
        </form>
      </div>  
        <div class="file-field input-field">
            <div class="btn">
                <span>Прикрепить файл</span>
                <input type="file" name="file" id="predictionFile">
            </div>
            <div class="file-path-wrapper">
                <input class="file-path validate" type="text" placeholder="Загрузить">
            </div>
        </div>
    </form>`;

    Swal.fire({
        title: 'Загрузить предсказание',
        html: uploadAvaForm,
        confirmButtonText: 'Отправить',
        showCancelButton: true,
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                let form = document.getElementById('uploadResearchForm');
                let formData = new FormData(form);
                let fileInput = document.getElementById("predictionFile");
                let mask_name = $("#mask_name").val();
                let mask_description = $("#mask_description").val();


                if (fileInput.files.length == 0) {
                    reject("Вы не прикрепили файл");
                }
                formData.append('file', fileInput.files[0]);
                formData.append("mask_name", mask_name);
                formData.append("mask_description", mask_description);
                formData.append("series_id", SERIES_ID);

                let xhr = new XMLHttpRequest();
                xhr.open('POST', form.getAttribute('action'), true);
                xhr.setRequestHeader("X-CSRFToken", token);

                xhr.responseType = 'json';

                xhr.addEventListener("load", function(){
                	followMaskProcessProgress(this.response);
                });

                xhr.send(formData);
            }).then(value => {
                console.log("VALUE: ", value)
            }, reason => {
                Swal.showValidationMessage (
                    `Ошибка: ${reason}`
                )
            })
        },
        allowOutsideClick: () => !Swal.isLoading()
    }).then((result) => {
       
    })
}