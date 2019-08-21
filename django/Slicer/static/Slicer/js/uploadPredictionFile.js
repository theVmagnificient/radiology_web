let TMP_RESULT;

function uploadPredictionFileForm() {
    let uploadAvaForm = '<form id = "uploadAvaForm" action="/account/uploadAva" enctype="multipart/form-data">' +
        '<div class="file-field input-field">' +
            '<div class="btn">' +
                '<span>Прикрепить файл</span>' +
                '<input type="file" name="file" id="fileInput">' +
            '</div>' +
            '<div class="file-path-wrapper">' +
                '<input class="file-path validate" type="text" placeholder="">' +
            '</div>' +
        '</div>' +
    '</form>';

    Swal.fire({
        title: 'Загрузить файл с предсказанием',
        html: uploadAvaForm,
        confirmButtonText: 'Отправить',
        showCancelButton: true,
        showLoaderOnConfirm: true,
        preConfirm: () => {
            var send = new Promise((resolve, reject) => {
                // let form = document.getElementById('uploadAvaForm');
                let file = document.getElementById("fileInput").files[0];

                var reader = new FileReader();
                reader.onload = function () {
                    let data = Papa.parse(reader.result).data;
                    let firstLocX = data[1][4];
                    let firstLocY = data[1][3];
                    let firstDiamX = data[1][7];
                    let firstDiamY = data[1][6];
                    drawSquare(firstLocX, firstLocY, firstDiamX, firstDiamY);
                };
                reader.readAsBinaryString(fileInput.files[0]);
            }).then((result) => {
                if (result.value) {
                    console.log("asd")
                }
          });
        }
    });
}

function drawSquare(locX, locY, diamX, diamY) {
    console.log(locX, locY, diamX, diamY)
    var c = document.getElementById("predictionCanvas");
    $("#predictionImage").fadeIn()
    var ctx = c.getContext("2d");

    base_image = new Image();
    base_image.src =  document.getElementById("imageLink").href;

    c.width = base_image.width;
    c.height = base_image.height;

    base_image.onload = function() {
        ctx.drawImage(base_image, 0, 0);
        ctx.beginPath();
        ctx.arc(locX, locY, diamX, 0, 2 * Math.PI);
        ctx.closePath();

        ctx.lineWidth = diamX / 5;
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        ctx.stroke(); 
    }
}

$(window).load(function () {
    $('.popupCloseButton').click(function(){
        $('#predictionImage').hide();
    });
});


// drawSquare(299.0, 374.0, 12.097405564859605, 12.097405564859605);