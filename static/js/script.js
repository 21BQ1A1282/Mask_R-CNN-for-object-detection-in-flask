// Logout functionality
document.getElementById('logout-button').addEventListener('click', function (e) {
    e.preventDefault();
    fetch('/logout', { method: 'POST' })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url;
            }
        });
});

// Drag and drop functionality
function processDrop(event) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.src = e.target.result;
            img.onload = function () {
                const canvas = document.getElementById('picture_canvas');
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                alert("Image uploaded successfully!");
            };
        };
        reader.readAsDataURL(file);
    }
}

function processDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
}

// Upload button functionality
document.getElementById('file-input').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.src = e.target.result;
            img.onload = function () {
                const canvas = document.getElementById('picture_canvas');
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                alert("Image uploaded successfully!");
            };
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('upload-form').onsubmit = async function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const response = await fetch('/object-detection/', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        // Show the "View Detected Image" button
        document.getElementById('view-detected-button').style.display = 'block';
    } else {
        console.error('Failed to process image');
    }
};

document.getElementById('upload-form').onsubmit = async function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const response = await fetch('/object-detection/', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        // Show the "View Detected Image" button
        document.getElementById('view-detected-button').style.display = 'block';
    } else {
        console.error('Failed to process image');
    }
};