const el = document.querySelector(".pageWrapper");

var movementStrength = 15;
var height = movementStrength / $(window).height()
var width = movementStrength / $(window).width()

el.addEventListener("mousemove", (e) => {
	var pageX = e.pageX - ($(window).width() / 2);
    var pageY = e.pageY - ($(window).height() / 2);
    var newvalueX = width * pageX * -1 - 25;
    var newvalueY = height * pageY * -1 - 50;
  	el.style.setProperty('--x', newvalueX  + "px");
  	el.style.setProperty('--y', newvalueY + "px");
});
