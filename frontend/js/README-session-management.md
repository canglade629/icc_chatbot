# Session Management System

This directory contains a comprehensive client-side session management system designed to minimize re-authentication and provide a seamless user experience.

## Files

### 1. `session-manager.js`
The core session management service that handles:
- **Token Management**: Automatic access token refresh before expiration
- **Session Persistence**: Stores session data in localStorage with automatic restoration
- **Activity Tracking**: Monitors user activity to extend session timeout
- **Background Refresh**: Automatically refreshes tokens in the background
- **Event System**: Provides events for session state changes

#### Key Features:
- **Automatic Token Refresh**: Refreshes tokens 5 minutes before expiry
- **Session Timeout**: 30-minute inactivity timeout (configurable)
- **Activity Tracking**: Monitors mouse, keyboard, and touch events
- **Graceful Degradation**: Falls back to manual token management if session manager fails
- **Event-Driven**: Emits events for session changes

#### Usage:
```javascript
// Check if user is authenticated
if (window.sessionManager.isUserAuthenticated()) {
    // User is logged in
}

// Make authenticated requests
const response = await window.sessionManager.authenticatedRequest('/api/endpoint', {
    method: 'POST',
    body: JSON.stringify(data)
});

// Listen for session events
window.sessionManager.on('sessionExpired', () => {
    // Handle session expiration
});
```

### 2. `session-timeout-warning.js`
Provides user-friendly warnings when sessions are about to expire:
- **Progressive Warnings**: Shows warnings at 5 minutes and 1 minute before expiry
- **Interactive UI**: Allows users to extend their session
- **Countdown Timer**: Shows remaining time
- **Auto-hide**: Automatically hides warnings on user activity

### 3. `session-status-indicator.js`
A subtle status indicator that shows:
- **Session Status**: Visual indicator of session health
- **Time Remaining**: Shows time until session expires (only when < 10 minutes)
- **Color Coding**: Green (active), Yellow (expiring soon), Red (critical)

## Configuration

### Session Manager Settings
```javascript
// In session-manager.js
this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
this.refreshBuffer = 5 * 60 * 1000;   // 5 minutes before expiry
```

### Backend Token Settings
```python
# In auth_service.py
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30    # 30 days
```

## Events

The session manager emits the following events:

- `sessionExpired`: Fired when session expires due to inactivity
- `tokenRefreshed`: Fired when tokens are successfully refreshed
- `userActivity`: Fired when user activity is detected
- `logout`: Fired when user logs out

## Integration

The session management system is automatically integrated into the main application:

1. **Authentication**: Login/signup automatically sets up session management
2. **API Calls**: All API calls use the session manager for authentication
3. **Page Load**: Session is automatically restored on page reload
4. **Logout**: Properly cleans up session data

## Benefits

1. **Reduced Re-authentication**: Users stay logged in longer with automatic token refresh
2. **Better UX**: Seamless experience with minimal interruptions
3. **Security**: Proper token management with secure refresh mechanisms
4. **Reliability**: Graceful handling of network issues and token expiration
5. **User Awareness**: Clear warnings and status indicators

## Browser Compatibility

- Modern browsers with localStorage support
- ES6+ features (arrow functions, classes, async/await)
- Event handling and DOM manipulation

## Security Considerations

- Tokens are stored in localStorage (consider httpOnly cookies for production)
- Automatic token refresh prevents token exposure
- Session timeout prevents indefinite sessions
- Activity tracking ensures sessions are only extended for active users
