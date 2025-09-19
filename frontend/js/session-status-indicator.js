/**
 * Session Status Indicator Component
 * Shows session status and time remaining
 */
class SessionStatusIndicator {
    constructor(sessionManager) {
        this.sessionManager = sessionManager;
        this.indicator = null;
        this.updateInterval = null;
        this.isVisible = false;
        
        this.init();
    }

    init() {
        this.createIndicator();
        this.startUpdateInterval();
        
        // Listen for session events
        this.sessionManager.on('tokenRefreshed', () => {
            this.updateStatus();
        });
        
        this.sessionManager.on('userActivity', () => {
            this.updateStatus();
        });
        
        this.sessionManager.on('logout', () => {
            this.hideIndicator();
        });
    }

    createIndicator() {
        this.indicator = document.createElement('div');
        this.indicator.id = 'session-status-indicator';
        this.indicator.className = 'fixed bottom-4 right-4 bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg text-sm z-40 hidden';
        this.indicator.innerHTML = `
            <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-green-500 rounded-full" id="status-dot"></div>
                <span id="status-text">Session Active</span>
                <span id="time-remaining" class="text-gray-300"></span>
            </div>
        `;
        
        document.body.appendChild(this.indicator);
    }

    showIndicator() {
        if (this.indicator) {
            this.indicator.classList.remove('hidden');
            this.isVisible = true;
        }
    }

    hideIndicator() {
        if (this.indicator) {
            this.indicator.classList.add('hidden');
            this.isVisible = false;
        }
    }

    startUpdateInterval() {
        // Update every 30 seconds
        this.updateInterval = setInterval(() => {
            this.updateStatus();
        }, 30000);
    }

    updateStatus() {
        if (!this.sessionManager.isUserAuthenticated()) {
            this.hideIndicator();
            return;
        }

        const sessionInfo = this.sessionManager.getSessionInfo();
        const timeUntilExpiry = this.sessionManager.getTimeUntilSessionExpiry();
        const minutes = Math.ceil(timeUntilExpiry / 60000);
        
        if (timeUntilExpiry <= 0) {
            this.hideIndicator();
            return;
        }

        // Show indicator if session is about to expire (within 10 minutes)
        if (timeUntilExpiry <= 10 * 60 * 1000) {
            this.showIndicator();
            
            const statusDot = this.indicator.querySelector('#status-dot');
            const statusText = this.indicator.querySelector('#status-text');
            const timeRemaining = this.indicator.querySelector('#time-remaining');
            
            if (timeUntilExpiry <= 5 * 60 * 1000) {
                // Less than 5 minutes - show warning
                statusDot.className = 'w-2 h-2 bg-yellow-500 rounded-full';
                statusText.textContent = 'Session Expiring Soon';
                timeRemaining.textContent = `${minutes}m`;
            } else {
                // 5-10 minutes - show info
                statusDot.className = 'w-2 h-2 bg-blue-500 rounded-full';
                statusText.textContent = 'Session Active';
                timeRemaining.textContent = `${minutes}m left`;
            }
        } else {
            this.hideIndicator();
        }
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        if (this.indicator) {
            this.indicator.remove();
            this.indicator = null;
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (window.sessionManager) {
            new SessionStatusIndicator(window.sessionManager);
        }
    });
} else {
    if (window.sessionManager) {
        new SessionStatusIndicator(window.sessionManager);
    }
}
