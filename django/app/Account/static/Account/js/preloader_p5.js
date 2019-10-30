window.addEventListener("load", function() {
    const loader = document.querySelector(".loader")
    loader.className += " hidden";
});

let cnv;
let stars = new Array();
let starsCnt = 250;
let speed;
let minSpeed = 5;
let maxSpeed = 30;

class Star {
	constructor() {
		this.x = random(-width, width);
		this.y = random(-height, height);
		this.z = random(width);
	}

	update() {
		this.z = this.z - speed;
		if (this.z < 1) {
			this.z = width;	 
			this.x = random(-width, width);
			this.y = random(-height, height);
		}
	}

	show() {
		fill(255);
		noStroke();

		let sx = map(this.x / this.z, 0, 1, 0, width);
		let sy = map(this.y / this.z, 0, 1, 0, height);

		let r = map(this.z, 0, width, 14, 0);

		ellipse(sx, sy, r, r);
	}
}

function centerCanvas() {
    var x = (windowWidth - width) / 2;
    var y = (windowHeight - height) / 2;
    cnv.position(x, y);
 }  

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}

function setup() {
	cnv = createCanvas(windowWidth, windowHeight);
    cnv.style('z-index', '900')
    centerCanvas();
    cnv.parent('loader');

    for (let i = 0; i < starsCnt; i++) {
    	stars.push(new Star());
    }
}

function draw() {
	speed = map(mouseX, 0, width, minSpeed, maxSpeed);
	background(0);
	translate(width / 2, height / 2);
	textSize(32);
	textAlign(CENTER, CENTER);
	for (let i = 0; i < stars.length; i++) {
		stars[i].update();
		stars[i].show();

		
		text("Находится в разработке", 10, 10);
	}
}