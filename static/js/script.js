// ============================================================
// ELEMENTS
// ============================================================

const imageInput =
    document.getElementById(
        "imageInput"
    );


const browseButton =
    document.getElementById(
        "browseButton"
    );


const dropArea =
    document.getElementById(
        "dropArea"
    );


const resultImage =
    document.getElementById(
        "resultImage"
    );


const noImage =
    document.getElementById(
        "noImage"
    );


const ageNumber =
    document.getElementById(
        "ageNumber"
    );


const ageGroup =
    document.getElementById(
        "ageGroup"
    );


const ageBar =
    document.getElementById(
        "ageBar"
    );


const ageValue =
    document.getElementById(
        "ageValue"
    );


const genderResult =
    document.getElementById(
        "genderResult"
    );


const completeMessage =
    document.getElementById(
        "completeMessage"
    );


// ============================================================
// BROWSE
// ============================================================

browseButton.addEventListener(
    "click",
    function () {

        imageInput.click();

    }
);


// ============================================================
// SELECT IMAGE
// ============================================================

imageInput.addEventListener(
    "change",
    function () {

        if (
            this.files &&
            this.files.length > 0
        ) {

            uploadImage(
                this.files[0]
            );

        }

    }
);


// ============================================================
// DRAG OVER
// ============================================================

dropArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropArea.classList.add(
            "dragover"
        );

    }
);


// ============================================================
// DRAG LEAVE
// ============================================================

dropArea.addEventListener(
    "dragleave",
    function () {

        dropArea.classList.remove(
            "dragover"
        );

    }
);


// ============================================================
// DROP
// ============================================================

dropArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropArea.classList.remove(
            "dragover"
        );


        const files =
            event.dataTransfer.files;


        if (
            files &&
            files.length > 0
        ) {

            uploadImage(
                files[0]
            );

        }

    }
);


// ============================================================
// UPLOAD
// ============================================================

function uploadImage(file) {


    // --------------------------------------------------------
    // FILE TYPE
    // --------------------------------------------------------

    const allowedTypes = [

        "image/jpeg",
        "image/jpg",
        "image/png"

    ];


    if (
        !allowedTypes.includes(
            file.type
        )
    ) {

        alert(
            "Please upload JPG, JPEG or PNG."
        );

        return;

    }


    // --------------------------------------------------------
    // FILE SIZE
    // --------------------------------------------------------

    if (
        file.size >
        5 * 1024 * 1024
    ) {

        alert(
            "Maximum image size is 5 MB."
        );

        return;

    }


    // --------------------------------------------------------
    // PREVIEW
    // --------------------------------------------------------

    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            resultImage.src =
                event.target.result;

            resultImage.style.display =
                "block";

            noImage.style.display =
                "none";

        };


    reader.readAsDataURL(
        file
    );


    // --------------------------------------------------------
    // FORM DATA
    // --------------------------------------------------------

    const formData =
        new FormData();


    formData.append(
        "image",
        file
    );


    // --------------------------------------------------------
    // LOADING
    // --------------------------------------------------------

    ageNumber.textContent =
        "...";


    ageGroup.textContent =
        "Analyzing...";


    ageValue.textContent =
        "...";


    genderResult.textContent =
    "...";


    ageBar.style.width =
        "30%";


    completeMessage.style.display =
        "none";


    // --------------------------------------------------------
    // SEND TO FLASK
    // --------------------------------------------------------

    fetch(
        "/predict",
        {
            method: "POST",
            body: formData
        }
    )


    .then(
        response => {

            if (!response.ok) {

                throw new Error(
                    "Server error"
                );

            }

            return response.json();

        }
    )


    .then(
        data => {


            if (
                !data.success
            ) {

                alert(
                    data.message
                );

                resetResult();

                return;

            }


            // ------------------------------------------------
            // IMAGE
            // ------------------------------------------------

            resultImage.src =
                data.image_url;

            resultImage.style.display =
                "block";

            noImage.style.display =
                "none";


            // ------------------------------------------------
            // AGE
            // ------------------------------------------------

            ageNumber.textContent =
                data.rounded_age;


            // ------------------------------------------------
            // AGE GROUP
            // ------------------------------------------------

            ageGroup.textContent =
                data.age_group;


            // ------------------------------------------------
            // AGE BAR
            // ------------------------------------------------

            let percentage =
                (
                    data.estimated_age /
                    116
                ) * 100;


            percentage =
                Math.max(
                    5,
                    Math.min(
                        100,
                        percentage
                    )
                );


            ageBar.style.width =
                percentage + "%";


            // ------------------------------------------------
            // EXACT AGE
            // ------------------------------------------------

            ageValue.textContent =
                data.estimated_age.toFixed(
                    2
                );


            // ------------------------------------------------
            // GENDER RESULT
            // ------------------------------------------------

            genderResult.textContent =
                data.gender;


            // ------------------------------------------------
            // COMPLETE
            // ------------------------------------------------

            completeMessage.style.display =
                "flex";

        }
    )


    .catch(
        error => {

            console.error(
                error
            );


            alert(
                "Something went wrong while predicting."
            );


            resetResult();

        }
    );

}


// ============================================================
// SAMPLE IMAGE
// ============================================================

function selectSample(
    filename
) {


    const sampleUrl =
        "/static/samples/" +
        filename;


    fetch(
        sampleUrl
    )

    .then(
        response => {

            if (!response.ok) {

                throw new Error(
                    "Sample image not found."
                );

            }

            return response.blob();

        }
    )

    .then(
        blob => {


            const file =
                new File(
                    [blob],
                    filename,
                    {
                        type:
                            blob.type ||
                            "image/jpeg"
                    }
                );


            uploadImage(
                file
            );

        }
    )

    .catch(
        error => {

            console.error(
                error
            );


            alert(
                "Sample image could not be loaded."
            );

        }
    );

}


// ============================================================
// RESET
// ============================================================

function resetResult() {


    ageNumber.textContent =
        "--";


    ageGroup.textContent =
        "--";


    ageValue.textContent =
        "--";


    genderResult.textContent =
        "--";


    ageBar.style.width =
        "0%";


    completeMessage.style.display =
        "none";

}