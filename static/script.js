let currentDetections = [];
let currentImageBase = '';

document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Store detection data globally
        currentDetections = data.detections;
        currentImageBase = data.image_url.replace('.jpg', '');

        // Display the result image
        const resultImage = document.getElementById("result-image");
        resultImage.src = "/image/" + data.image_url;
        
        // Wait for image to load before setting up click handlers
        resultImage.onload = function() {
            setupImageClickHandlers(resultImage, data.detections);
        };

        // Display summary
        let summaryHTML = "<h3>Detections Summary</h3>";
        for (const [cls, count] of Object.entries(data.summary)) {
            summaryHTML += `<div>${cls}: ${count}</div>`;
        }

        // Add clickable detection list
        summaryHTML += "<h3>Click on objects to highlight them:</h3>";
        summaryHTML += "<div id='detection-list'>";
        data.detections.forEach((detection, index) => {
            summaryHTML += `
                <div class="detection-item" data-detection-id="${detection.id}" style="
                    cursor: pointer; 
                    padding: 8px; 
                    margin: 4px 0; 
                    border: 2px solid rgb(${detection.color[0]}, ${detection.color[1]}, ${detection.color[2]}); 
                    border-radius: 4px;
                    background-color: rgba(${detection.color[0]}, ${detection.color[1]}, ${detection.color[2]}, 0.1);
                    transition: all 0.3s ease;
                ">
                    <strong>${detection.class}</strong> - ${(detection.conf * 100).toFixed(1)}% confidence
                </div>
            `;
        });
        summaryHTML += "</div>";

        document.getElementById("summary-box").innerHTML = summaryHTML;

        // Add click handlers to detection list items
        document.querySelectorAll('.detection-item').forEach(item => {
            item.addEventListener('click', function() {
                const detectionId = this.dataset.detectionId;
                highlightDetection(detectionId);
                
                // Visual feedback for clicked item
                document.querySelectorAll('.detection-item').forEach(i => i.style.transform = 'scale(1)');
                this.style.transform = 'scale(1.05)';
            });

            // Hover effects
            item.addEventListener('mouseenter', function() {
                this.style.backgroundColor = this.style.backgroundColor.replace('0.1', '0.3');
                this.style.transform = 'scale(1.02)';
            });

            item.addEventListener('mouseleave', function() {
                this.style.backgroundColor = this.style.backgroundColor.replace('0.3', '0.1');
                if (!this.classList.contains('selected')) {
                    this.style.transform = 'scale(1)';
                }
            });
        });

    } catch (error) {
        console.error('Error:', error);
        document.getElementById("summary-box").innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
    }
});

function setupImageClickHandlers(imageElement, detections) {
    // Remove any existing click handlers
    const existingOverlay = document.getElementById('image-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    // Create an overlay div for click detection
    const overlay = document.createElement('div');
    overlay.id = 'image-overlay';
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.cursor = 'pointer';
    overlay.style.zIndex = '10';

    // Position the overlay over the image
    const imageContainer = imageElement.parentElement;
    if (imageContainer.style.position !== 'relative') {
        imageContainer.style.position = 'relative';
    }
    imageContainer.appendChild(overlay);

    overlay.addEventListener('click', function(event) {
        const rect = imageElement.getBoundingClientRect();
        const scaleX = imageElement.naturalWidth / imageElement.clientWidth;
        const scaleY = imageElement.naturalHeight / imageElement.clientHeight;
        
        const clickX = (event.clientX - rect.left) * scaleX;
        const clickY = (event.clientY - rect.top) * scaleY;

        // Find which detection was clicked
        for (const detection of detections) {
            const [x1, y1, x2, y2] = detection.bbox;
            if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                highlightDetection(detection.id);
                
                // Update detection list visual state
                document.querySelectorAll('.detection-item').forEach(item => {
                    item.classList.remove('selected');
                    item.style.transform = 'scale(1)';
                });
                
                const clickedItem = document.querySelector(`[data-detection-id="${detection.id}"]`);
                if (clickedItem) {
                    clickedItem.classList.add('selected');
                    clickedItem.style.transform = 'scale(1.05)';
                }
                
                break;
            }
        }
    });

    // Add visual feedback on hover
    overlay.addEventListener('mousemove', function(event) {
        const rect = imageElement.getBoundingClientRect();
        const scaleX = imageElement.naturalWidth / imageElement.clientWidth;
        const scaleY = imageElement.naturalHeight / imageElement.clientHeight;
        
        const hoverX = (event.clientX - rect.left) * scaleX;
        const hoverY = (event.clientY - rect.top) * scaleY;

        let isOverDetection = false;
        for (const detection of detections) {
            const [x1, y1, x2, y2] = detection.bbox;
            if (hoverX >= x1 && hoverX <= x2 && hoverY >= y1 && hoverY <= y2) {
                isOverDetection = true;
                break;
            }
        }

        overlay.style.cursor = isOverDetection ? 'pointer' : 'default';
    });
}

async function highlightDetection(detectionId) {
    try {
        const resultImage = document.getElementById("result-image");
        
        // Show loading indicator
        resultImage.style.opacity = '0.7';
        
        // Fetch highlighted image
        const response = await fetch(`/highlight/${currentImageBase}/${detectionId}`);
        if (!response.ok) {
            throw new Error('Failed to highlight detection');
        }

        // Create blob URL for the highlighted image
        const blob = await response.blob();
        const highlightedImageUrl = URL.createObjectURL(blob);
        
        // Update the image source
        resultImage.src = highlightedImageUrl;
        resultImage.style.opacity = '1';
        
        // Clean up the blob URL after a delay to prevent memory leaks
        setTimeout(() => {
            URL.revokeObjectURL(highlightedImageUrl);
        }, 5000);

    } catch (error) {
        console.error('Error highlighting detection:', error);
        document.getElementById("result-image").style.opacity = '1';
    }
}

// Add double-click to reset to original image
document.addEventListener('DOMContentLoaded', function() {
    const resultImage = document.getElementById("result-image");
    if (resultImage) {
        resultImage.addEventListener('dblclick', function() {
            if (currentImageBase) {
                this.src = "/image/" + currentImageBase + ".jpg";
                
                // Reset detection list visual state
                document.querySelectorAll('.detection-item').forEach(item => {
                    item.classList.remove('selected');
                    item.style.transform = 'scale(1)';
                });
            }
        });
    }
});

