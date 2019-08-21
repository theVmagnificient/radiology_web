function searchResearch() {
	const url = "/home/searchResearch";
	let searchWord = $("#searchResearchInput").val();
	
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
            word: searchWord,
        },
        success: (function (response) {
            if (!response["ok"]) {
                alert("Похоже на сервере возникла ошибка");
            } else {
                renderResearches(response["res"]);
            }
        }),
        error: (function (response) {
        	console.log("ERROR: ", response);
        }),
    });
}

function renderResearches(json) {
    res = JSON.parse(json);
	if (res.length == 0) {
		$("#researchListContainer").html("<h1> Ничего не найдено </h1>");
	} else {
		console.log(res);
		let resCnt = 0;
		htmlContent = '<ul class="collapsible">';
		$.each(res, function(key, value) {
			resCnt++;
			htmlContent += `
				<li class = "show${resCnt} showBase">
                    <div class="collapsible-header"><i class="material-icons series_icon">local_see</i>
                        <p class="truncate"><b>${resCnt}. </b> ${value.fields.SeriesInstanceUID} </p>
                    </div>
                    <div class="collapsible-body">
                        <div class="row">
                            <div class="col s12 m6 series_info">
                                <p class="truncate"><b>Patient ID: </b> ${ value.fields.PatientID } </p>
                                <p class="truncate"><b>Study ID: </b> ${ value.fields.StudyID } </p>
                                <p><b>Слои:</b> ${ value.fields.slicesCnt } </p>
                                <br /><br /><a href="/series/view?id=${value.fields.seriesID}">Полный просмотр</a>
                            </div>
                            <div class="col s12 m6">
                                <img src='/media/images/${value.fields.slices_dir}/${value.fields.previewSlice}'
                                    class="series_preview" />
                            </div>
                        </div>
                    </div>
                </li>
			`;
		});
		htmlContent += '</ul>';
		$("#researchListContainer").html(htmlContent);
		reRender();
		initResearchList();
	}
}

$(document).ready(function() {
    $("#searchResearchInput").keyup(function(event) {
        // Enter
        if (event.keyCode === 13) {
            searchResearch();
        }

        // Nums
        if ((event.keyCode >= 48 && event.keyCode <= 57) ||
            (event.keyCode >= 96 && event.keyCode <= 105)) { 
            searchResearch();
        }
        // Dot
        if (event.keyCode === 190) {
            searchResearch();
        }

        // Letters
        if (event.keyCode >= 65 && event.keyCode <= 90) {
            searchResearch();
        }
        // Space, BackSpace, Delete
        if (event.keyCode == 32 || event.keyCode == 8 || event.keyCode == 46) {
            searchResearch();
        }
    });
});