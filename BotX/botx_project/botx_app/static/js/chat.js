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

function toggleChatbot() {
    const chatbotContainer = document.getElementById('chatbotContainer');
    chatbotContainer.style.display = chatbotContainer.style.display === 'none' ? 'flex' : 'none';
}

let isFirstContentQuery = true;

function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    if (message) {
        const chatbotBody = document.getElementById('chatbotBody');
        const typingIndicator = document.getElementById('typingIndicator');
        const warningText = document.getElementById('warningText');

        // Display user message
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.textContent = message;
        chatbotBody.appendChild(userMessage);

        // Clear input field
        userInput.value = '';

        // Scroll to the bottom
        setTimeout(() => {
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        }, 100); 

        // Show typing indicator inside chatbot body
        typingIndicator.textContent = "BotX is typing..."
        typingIndicator.style.display = 'block';
        chatbotBody.appendChild(typingIndicator);
        chatbotBody.scrollTop = chatbotBody.scrollHeight;

        // Hide warning text after the first content-based query
        if (isFirstContentQuery) {
            warningText.style.display = 'none';
            isFirstContentQuery = false;
        }

        // Send message to backend
        fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Server response:', data);
            // Hide typing indicator
            typingIndicator.style.display = 'none';
        
            // Display bot response
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.textContent = data.response || "No response from server";
            chatbotBody.appendChild(botMessage);
        
            // Scroll to the bottom
            setTimeout(() => {
                chatbotBody.scrollTop = chatbotBody.scrollHeight;
            }, 100);
        })
        .catch(error => {
            console.error('Error:', error);
            // Hide typing indicator
            typingIndicator.style.display = 'none';
        
            // Display error message
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot';
            botMessage.textContent = "Sorry, something went wrong.";
            chatbotBody.appendChild(botMessage);
        
            // Scroll to the bottom
            setTimeout(() => {
                chatbotBody.scrollTop = chatbotBody.scrollHeight;
            }, 100);
        });
    }
}
document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent form submission (if inside a form)
        sendMessage();
    }
});