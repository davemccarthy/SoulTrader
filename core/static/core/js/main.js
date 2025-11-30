// CSRF token helper
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

// AJAX utility function
async function ajaxRequest(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        return await response.json();
    } catch (error) {
        console.error('AJAX request failed:', error);
        return null;
    }
}

// Portfolio widget refresh (to be implemented)
async function refreshPortfolioWidget() {
    // Placeholder - will fetch and update portfolio stats
    console.log('Portfolio widget refresh - to be implemented');
}

// Dynamic ticker animation duration based on content size
function adjustTickerSpeed() {
    const tickerText = document.querySelector('.ticker-text');
    if (!tickerText) return;
    
    // Measure the actual width of the content (first half, since it's duplicated)
    const contentWidth = tickerText.scrollWidth / 2;
    const containerWidth = tickerText.parentElement.offsetWidth;
    
    // Desired scroll speed in pixels per second (adjust this to control speed)
    const pixelsPerSecond = 50; // Lower = slower, Higher = faster
    
    // Calculate duration: time = distance / speed
    // We need to scroll from 0 to -50% (half the content width)
    const scrollDistance = contentWidth + containerWidth; // Full width + container width
    const duration = scrollDistance / pixelsPerSecond;
    
    // Set minimum and maximum duration bounds (optional)
    const minDuration = 20; // Minimum 20 seconds
    const maxDuration = 120; // Maximum 120 seconds
    const finalDuration = Math.max(minDuration, Math.min(maxDuration, duration));
    
    // Apply the duration to the animation
    tickerText.style.animationDuration = `${finalDuration}s`;
}

// Run on page load
document.addEventListener('DOMContentLoaded', adjustTickerSpeed);

// Also run after a short delay to ensure content is fully rendered
setTimeout(adjustTickerSpeed, 100);

