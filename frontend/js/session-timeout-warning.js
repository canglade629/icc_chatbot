/**
 * Session Timeout Warning Component
 * Shows warnings when session is about to expire
 */
class SessionTimeoutWarning {
    constructor(sessionManager) {
        this.sessionManager = sessionManager;
        this.warningModal = null;
        this.countdownInterval = null;
        this.warningShown = false;
        this.warningThreshold = 5 * 60 * 1000; // 5 minutes before expiry
        this.finalWarningThreshold = 1 * 60 * 1000; // 1 minute before expiry
        
        this.init();
    }

    init() {
        // Check for session expiry every 30 seconds
        setInterval(() => {
            this.checkSessionStatus();
        }, 30000);

        // Listen for session events
        this.sessionManager.on('sessionExpired', () => {
            this.hideWarning();
            this.showSessionExpiredModal();
        });

        this.sessionManager.on('userActivity', () => {
            if (this.warningShown) {
                this.hideWarning();
            }
        });
    }

    checkSessionStatus() {
        if (!this.sessionManager.isUserAuthenticated()) {
            return;
        }

        const timeUntilExpiry = this.sessionManager.getTimeUntilSessionExpiry();
        
        if (timeUntilExpiry <= this.finalWarningThreshold && !this.warningShown) {
            this.showFinalWarning(timeUntilExpiry);
        } else if (timeUntilExpiry <= this.warningThreshold && !this.warningShown) {
            this.showWarning(timeUntilExpiry);
        }
    }

    showWarning(timeUntilExpiry) {
        this.warningShown = true;
        this.createWarningModal('warning', timeUntilExpiry);
    }

    showFinalWarning(timeUntilExpiry) {
        this.warningShown = true;
        this.createWarningModal('final', timeUntilExpiry);
    }

    createWarningModal(type, timeUntilExpiry) {
        // Remove existing modal
        this.hideWarning();

        const isFinal = type === 'final';
        const minutes = Math.ceil(timeUntilExpiry / 60000);
        
        this.warningModal = document.createElement('div');
        this.warningModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        this.warningModal.innerHTML = `
            <div class="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 ${isFinal ? 'border-l-4 border-red-500' : 'border-l-4 border-yellow-500'}">
                <div class="p-6">
                    <div class="flex items-center mb-4">
                        <div class="flex-shrink-0">
                            <i class="fas ${isFinal ? 'fa-exclamation-triangle text-red-500' : 'fa-clock text-yellow-500'} text-2xl"></i>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-lg font-medium text-gray-900">
                                ${isFinal ? 'Session About to Expire' : 'Session Timeout Warning'}
                            </h3>
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <p class="text-sm text-gray-600">
                            ${isFinal 
                                ? `Your session will expire in <span id="countdown-timer" class="font-bold text-red-600">${minutes}</span> minute${minutes !== 1 ? 's' : ''}. Please save your work and refresh the page to continue.`
                                : `Your session will expire in <span id="countdown-timer" class="font-bold text-yellow-600">${minutes}</span> minute${minutes !== 1 ? 's' : ''}. Click "Extend Session" to continue working.`
                            }
                        </p>
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        ${!isFinal ? `
                            <button id="extend-session-btn" class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                Extend Session
                            </button>
                        ` : ''}
                        <button id="refresh-page-btn" class="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500">
                            ${isFinal ? 'Refresh Page' : 'Refresh Now'}
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.warningModal);

        // Add event listeners
        if (!isFinal) {
            const extendBtn = this.warningModal.querySelector('#extend-session-btn');
            extendBtn.addEventListener('click', () => {
                this.extendSession();
            });
        }

        const refreshBtn = this.warningModal.querySelector('#refresh-page-btn');
        refreshBtn.addEventListener('click', () => {
            window.location.reload();
        });

        // Start countdown
        this.startCountdown(timeUntilExpiry);

        // Auto-hide after 30 seconds if not final warning
        if (!isFinal) {
            setTimeout(() => {
                if (this.warningShown) {
                    this.hideWarning();
                }
            }, 30000);
        }
    }

    startCountdown(timeUntilExpiry) {
        const countdownElement = this.warningModal.querySelector('#countdown-timer');
        if (!countdownElement) return;

        let remainingTime = timeUntilExpiry;
        
        this.countdownInterval = setInterval(() => {
            remainingTime -= 1000;
            const minutes = Math.ceil(remainingTime / 60000);
            
            if (remainingTime <= 0) {
                clearInterval(this.countdownInterval);
                this.countdownInterval = null;
                return;
            }
            
            countdownElement.textContent = minutes;
        }, 1000);
    }

    extendSession() {
        this.sessionManager.extendSession();
        this.hideWarning();
        
        // Show success message
        this.showToast('Session extended successfully', 'success');
    }

    hideWarning() {
        if (this.warningModal) {
            this.warningModal.remove();
            this.warningModal = null;
        }
        
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
        
        this.warningShown = false;
    }

    showSessionExpiredModal() {
        const expiredModal = document.createElement('div');
        expiredModal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        expiredModal.innerHTML = `
            <div class="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 border-l-4 border-red-500">
                <div class="p-6">
                    <div class="flex items-center mb-4">
                        <div class="flex-shrink-0">
                            <i class="fas fa-exclamation-circle text-red-500 text-2xl"></i>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-lg font-medium text-gray-900">Session Expired</h3>
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <p class="text-sm text-gray-600">
                            Your session has expired due to inactivity. Please log in again to continue.
                        </p>
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button id="login-again-btn" class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            Log In Again
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(expiredModal);

        const loginBtn = expiredModal.querySelector('#login-again-btn');
        loginBtn.addEventListener('click', () => {
            window.location.href = '/auth.html';
        });
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const bgColor = type === 'success' ? 'bg-green-500' : type === 'error' ? 'bg-red-500' : 'bg-blue-500';
        
        toast.className = `fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg z-50`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (window.sessionManager) {
            new SessionTimeoutWarning(window.sessionManager);
        }
    });
} else {
    if (window.sessionManager) {
        new SessionTimeoutWarning(window.sessionManager);
    }
}
