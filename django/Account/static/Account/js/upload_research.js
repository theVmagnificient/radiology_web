let uploadContainer = $("#uploadResearchContainer");
let loc = window.location;

let endpoint = (loc.protocol == "https:" ? "ws://" : "ws://") + loc.host + "/uploadResearch/";
let socket;

let timeoutWebsocketReqID;

function showError() {

}

function startDelayedWebsocketRequest(socketData) {
	timeoutWebsocketReqID = window.setInterval(function() {
		socket.send(JSON.stringify(socketData));
	}, 1000);
}

function stopDelayedWebsocketRequest() {
	window.clearInterval(timeoutWebsocketReqID);
}

function followFile(response) {
	if (response["filename"] == null || !response["ok"]) {
		renderLoadingResearche("Ошибка при загрузке файла. Повторите попытку позже.", 0);
		return;
	}
	
	filename = response["filename"]
	renderLoadingResearche("Файл загружен. Подключаемся к серверу...", -1);

	socket = new ReconnectingWebSocket(endpoint);
	socket.onopen = function(e) {
		renderLoadingResearche("Соединение установлено. Получение данных...", -1);
		let socketData = {
      		"filename": filename,
    	}
    	startDelayedWebsocketRequest(socketData);
	}
	socket.onmessage = function(e) {
		let data = JSON.parse(e["data"]);
		console.log(data);

		let progress = parseFloat(data["progress"])
		let roundedProgress = Math.round(progress)
		let status = parseInt(data["status"])

		if (status == 1) {
			renderLoadingResearche(`Извлекаем снимки из архива... `, -1);
		} else if (status == 2) {
			renderLoadingResearche(`Обработка исследования. ${roundedProgress}%`, progress);
        } else if (status == 3){
            renderLoadingResearche(`Работаем с базой данных...`, -1);
		} else if (status == 5) {
			renderLoadingResearche("Исследование обработано. Обновляем данные...", -1);
			stopDelayedWebsocketRequest()
			window.setTimeout(function() {
				window.location = "/home/view";
			}, 2000);
		}
		else {
			renderLoadingResearche("На сервере возникла ошибка.", -1);
		}

	}
	socket.onerror = function(e) {
		renderLoadingResearche("Возникла ошибка на сервере. Обработка не завершена.", 0);
		stopDelayedWebsocketRequest()
	}
	socket.onclose = function(e) {
		renderLoadingResearche("Соединение разорвано. Переподключаемся...", -1);
		stopDelayedWebsocketRequest()
	}
}

function renderLoadingResearche(status, progress=-1) {
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


function uploadResearch() {
    const url = '/home/uploadResearch';
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    let uploadAvaForm = `<form id = "uploadResearchForm" action="/home/uploadResearch" enctype="multipart/form-data">'
        <div class="file-field input-field">
            <div class="btn">
                <span>Загрузите архив</span>
                <input type="file" name="file" id="file">
            </div>
            <div class="file-path-wrapper">
                <input class="file-path validate" type="text" placeholder="Загрузить">
            </div>
        </div>
    </form>`;

    Swal.fire({
        title: 'Загрузить исследование',
        html: uploadAvaForm,
        confirmButtonText: 'Отправить',
        showCancelButton: true,
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                let form = document.getElementById('uploadResearchForm');
                let formData = new FormData(form);

                if (file.value.length == 0) {
                    reject("Вы не прикрепили файл")
                }
				renderLoadingResearche("Загрузка файла на сервер", -1);
                formData.append('file', file);
                
                let xhr = new XMLHttpRequest();
                xhr.open('POST', form.getAttribute('action'), true);
                xhr.setRequestHeader("X-CSRFToken", token);

                xhr.responseType = 'json';

                xhr.addEventListener("load", function(){
                	followFile(this.response);
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