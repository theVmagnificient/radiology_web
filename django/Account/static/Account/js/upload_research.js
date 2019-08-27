let uploadContainer = $("#uploadResearchContainer");
let loc = window.location;

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
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    let uploadAvaForm = `<form id = "uploadResearchForm" action="/series/upload_research" enctype="multipart/form-data">'
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

				xhr.upload.addEventListener("progress", function(evt){
					if (evt.lengthComputable) {
						var percentComplete = (evt.loaded / evt.total) * 100;
						renderLoadingResearche("Загрузка файла на сервер", percentComplete);
						if (percentComplete == 100) {
							renderLoadingResearche("Обработка исследования...");
						}
					}
			   }, false);

			   xhr.onreadystatechange = function() {
					if(xhr.readyState === 4 && xhr.status === 200) {
						let resp = xhr.response;
						if (resp.ok) {
							window.location = "/home/view";
						} else {
							Swal.fire({
								type: 'error',
								title: 'Ошибка',
								text: resp.error,
							});
							uploadContainer.html(" ");
						}
				  	}
				}

                xhr.responseType = 'json';
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