$(document).ready(() => {
        let i = 0;
        let timer = setInterval(() => {
                i++;
                if (i == 10)
                    clearInterval(timer);
                else {
                    // console.log($(`.show${i}`));
                    if ($(`.show${i}`).hasClass('showNavbar'))
                        $(`.show${i}`).css('display', 'flex').hide().fadeIn();
                    else
                        $(`.show${i}`).fadeIn(700);
          	}
     	}, 200);
 });

function reRender() {
    let i = 0;
        let timer = setInterval(() => {
                i++;
                if (i == 10)
                    clearInterval(timer);
                else {
                    // console.log($(`.show${i}`));
                    if ($(`.show${i}`).hasClass('showNavbar'))
                        $(`.show${i}`).css('display', 'flex').hide().fadeIn();
                    else
                        $(`.show${i}`).fadeIn(700);
            }
    }, 200);
}