document.addEventListener("DOMContentLoaded", function () {
    const inputType = document.getElementById("inputType");
    const videoFile = document.getElementById("videoFile");
    const videoLabel = document.getElementById("videoLabel");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const statusMessage = document.getElementById("statusMessage");

    inputType.addEventListener("change", function () {
        if (inputType.value === "video") {
            videoFile.style.display = "block";
            videoLabel.style.display = "block";
        } else {
            videoFile.style.display = "none";
            videoLabel.style.display = "none";
        }
    });

    startBtn.addEventListener("click", function () {
        let input_type = inputType.value;
        let video_file = videoFile.files.length > 0 ? videoFile.files[0].name : null;

        fetch('/start-detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input_type: input_type,
                video_file: video_file
            })
        }).then(response => response.json())
          .then(data => {
              statusMessage.textContent = data.message;
          });
    });

    stopBtn.addEventListener("click", function () {
        fetch('/stop-detection', {
            method: 'POST'
        }).then(response => response.json())
          .then(data => {
              statusMessage.textContent = data.message;
          });
    });
});
