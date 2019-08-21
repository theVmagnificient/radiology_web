function sendForm() {

    let responseContainer = document.getElementById("response")
    let login = document.getElementById("login").value
    let password = document.getElementById("password").value
    let token = document.getElementsByName("csrfmiddlewaretoken")[0].getAttribute("value")

    const url = '/auth/login/';
    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", token);
            }
        }
    });

    let res;
    $.ajax({
        type: "POST",
        url: url,
        async: false,
        data: {
            login: login,
            passwd: password,
        },
        success: (function(response) {
            res = response;
        }),
        error: (function(response) {
            res = response;
        }),
    });
    return res;
}

(function ($) {
    "use strict";


    /*==================================================================
    [ Focus Contact2 ]*/
    $('.input100').each(function(){
        $(this).on('blur', function(){
            if($(this).val().trim() != "") {
                $(this).addClass('has-val');
            }
            else {
                $(this).removeClass('has-val');
            }
        })    
    })
  
  
    /*==================================================================
    [ Validate ]*/
    var input = $('.validate-input .input100');

    $('.validate-form').on('submit',function(){
        var check = true;

        for(var i=0; i<input.length; i++) {
            let v = validate(input[i])
            if(v != null){
                showValidate(input[i], v);
                check=false;
            }
        }

        if (check) {
            let response = sendForm();
            if (!response['success']) {
                showValidate($("#login"), response['message']);
            }
            if (response['redirect']) {
                location.href = response['redirect'];
            }
        }
    });


    $('.validate-form .input100').each(function(){
        $(this).focus(function(){
           hideValidate(this);
        });
    });

    function validate (input) {
        if($(input).attr('name') == 'login') {
            if($(input).val().trim().match(/^[a-zA-Z0-9]+$/) == null) {
                return "Поле содержит запрещенные символы"
            }
        } else {
            if($(input).val().trim() == ''){
                return "Это поле обязательное";
            }
        }
        return null;
    }

    function showValidate(input, msg="") {
        var thisAlert = $(input).parent();

        if (msg != "") {
            $(thisAlert).attr("data-validate", msg);
        }

        $(thisAlert).addClass('alert-validate');
    }

    function hideValidate(input, msg="") {
        var thisAlert = $(input).parent();

        $(thisAlert).attr("data-validate", "");
        $(thisAlert).removeClass('alert-validate');
    }
    
    $(".login100-more").mouseover(function(){

    });
    $(".login100-more").mouseout(function(){
    });

})(jQuery);