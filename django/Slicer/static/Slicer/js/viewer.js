slider.addEventListener('mouseout', function () {
  getSlice(slider.noUiSlider.get());
});
slider.addEventListener('click', function () {
  getSlice(slider.noUiSlider.get());
});
slider.addEventListener('mousemove', function () {
  getSlice(slider.noUiSlider.get());
});

$("#changePallete").change(function() {
  var str = "";
  $( "#changePallete option:selected" ).each(function() {
    str = $( this ).text();
  });
  if (str == "Стандартная") {
    DICOM_PALLETE = "";
  } else {
    DICOM_PALLETE = str;
  }
  getSlice(slider.noUiSlider.get());
});

$('#isPreview').click(function() {
  const url = "/series/setPreview";
  let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value");

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
  $.ajax({
      type: "POST",
      url: url,
      data: {
          id: SERIES_ID,
          fileName: imageName,
      },
      success: (function (response) {
          console.log(response);
          if (!response["ok"]) {
            M.toast({html: response["msg"]});
          } else {
            M.toast({html: "Сохранено"});
            previewSlice = imageName;
          }
      }),
  });
});