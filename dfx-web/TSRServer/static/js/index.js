const dropArea = document.getElementById("dragArea");
const mainArea = document.getElementById("mainArea");
const fileInput = document.getElementById("fileInput");
const containerTool = document.getElementById("containerTool");
const containerUpload = document.getElementById("containerUpload");
const containerProcess = document.getElementById("containerProcess");
const csrftoken = document.querySelector("meta[name=csrf_token]").content
let durl;


allowFormat = ["pdf", "png", "jpg", "jpeg"]

mainArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.display = "block";
});

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.display = "block";
});

dropArea.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropArea.style.display = "none";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.display = "none";
    const data = e.dataTransfer;
    if (!isValid(data)) return;
    fileInput.files = data.files;
    fileUpload();
});

fileInput.addEventListener("change", (e) => {
    fileUpload();
})

const isValid = (data) => {
    if (data.types.indexOf('Files') < 0) return false;
    if (data.files.length > 1) {
        return false
    }
    let fileFormat = data.files[0].name.split(".").reverse()[0].toLowerCase();
    if (allowFormat.indexOf(fileFormat) < 0) {
        alert("파일 형식을 지원하지 않습니다.")
        return false
    }
    return true
}

const fileUpload = () => {
    containerTool.style.display = "none";
    containerUpload.style.display = "block"
    let formData = new FormData();
    formData.append("uploadFile", fileInput.files[0])
    $("#uploadFileName").text(fileInput.files[0].name);

    $.ajax({
        url: "/upload",
        type: "POST",
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        headers: {"X-CSRFToken": csrftoken},
        xhr: function () {
            let xhr = $.ajaxSettings.xhr();
            xhr.upload.onprogress = function (event) {
                let perc = Math.round((event.loaded / event.total) * 100);
                $("#uploadPerc").text(perc + "%");
                $("#uploadBar").val(perc);
            };
            return xhr;
        },
        success: function (res) {
            containerUpload.style.display = "none"
            containerProcess.style.display = "block"
            console.log(res["url"])
            durl = res["url"];
            setInterval(checkComplete, 5000);
        }

    })
}

const checkComplete = () => {
    $.ajax({
        url: "/check",
        type: "get",
        data: {"url": durl},
        success: function (res) {
            if (res["code"] === 200) {
                location.href = "/download?code="+durl;
            } else if (res["code"] === 201){
                console.log("아직");
            }
        }
    })
}