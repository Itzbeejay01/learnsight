/**
 * LearnSight - Student Academic Performance Prediction System
 * JavaScript Interactivity
 * Version: 1.0.0
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Initialize Bootstrap Tooltips
    // ═══════════════════════════════════════════════════════════════════════════════
    
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            placement: 'top',
            trigger: 'hover focus',
            container: 'body'
        });
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Form Validation
    // ═══════════════════════════════════════════════════════════════════════════════
    
    const predictionForm = document.getElementById('predictionForm');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            let isValid = true;
            let firstError = null;
            
            // Get all required inputs
            const requiredInputs = predictionForm.querySelectorAll('input[required], select[required]');
            
            requiredInputs.forEach(function(input) {
                // Remove previous error styling
                input.classList.remove('is-invalid');
                
                // Check if empty
                if (!input.value.trim()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                    
                    if (!firstError) {
                        firstError = input;
                    }
                }
                
                // Validate number ranges
                if (input.type === 'number' && input.value) {
                    const value = parseFloat(input.value);
                    const min = parseFloat(input.min);
                    const max = parseFloat(input.max);
                    
                    if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
                        isValid = false;
                        input.classList.add('is-invalid');
                        
                        if (!firstError) {
                            firstError = input;
                        }
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                
                // Scroll to first error
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    firstError.focus();
                }
                
                // Show error message
                showNotification('Please fill in all required fields correctly.', 'error');
            } else {
                // Show loading overlay
                showLoadingOverlay();
            }
        });
        
        // Real-time validation feedback
        const inputs = predictionForm.querySelectorAll('input, select');
        inputs.forEach(function(input) {
            input.addEventListener('blur', function() {
                validateInput(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid')) {
                    validateInput(this);
                }
            });
        });
    }
    
    function validateInput(input) {
        // Remove previous error styling
        input.classList.remove('is-invalid');
        
        // Check if required and empty
        if (input.hasAttribute('required') && !input.value.trim()) {
            input.classList.add('is-invalid');
            return false;
        }
        
        // Validate number ranges
        if (input.type === 'number' && input.value) {
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
                input.classList.add('is-invalid');
                return false;
            }
        }
        
        return true;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Loading Overlay
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function showLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('active');
            
            // Update loading message periodically
            const messages = [
                'LearnSight is analyzing student data...',
                'Processing machine learning models...',
                'Generating SHAP explanations...',
                'Compiling personalized insights...',
                'Almost there...'
            ];
            
            let messageIndex = 0;
            const messageElement = overlay.querySelector('p');
            
            const messageInterval = setInterval(function() {
                if (!overlay.classList.contains('active')) {
                    clearInterval(messageInterval);
                    return;
                }
                
                messageIndex = (messageIndex + 1) % messages.length;
                if (messageElement) {
                    messageElement.textContent = messages[messageIndex];
                }
            }, 2000);
        }
    }
    
    function hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }
    
    // Hide loading overlay on page load (in case of back button)
    hideLoadingOverlay();
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Smooth Scrolling
    // ═══════════════════════════════════════════════════════════════════════════════
    
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Notification System
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.learnsight-notification');
        existingNotifications.forEach(function(n) {
            n.remove();
        });
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `learnsight-notification alert alert-${type === 'error' ? 'danger' : type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 90px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            max-width: 500px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            animation: slideIn 0.3s ease;
        `;
        
        // Add animation styles
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                @keyframes slideOut {
                    from {
                        transform: translateX(0);
                        opacity: 1;
                    }
                    to {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(function() {
            notification.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(function() {
                notification.remove();
            }, 300);
        }, 5000);
    }
    
    // Make showNotification globally accessible
    window.showNotification = showNotification;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Form Reset Handler
    // ═══════════════════════════════════════════════════════════════════════════════
    
    const resetButtons = document.querySelectorAll('button[type="reset"]');
    resetButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            // Confirm reset
            if (!confirm('Are you sure you want to clear all form data?')) {
                e.preventDefault();
                return;
            }
            
            // Remove invalid states
            const form = this.closest('form');
            if (form) {
                const invalidInputs = form.querySelectorAll('.is-invalid');
                invalidInputs.forEach(function(input) {
                    input.classList.remove('is-invalid');
                });
            }
        });
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Number Input Enhancement
    // ═══════════════════════════════════════════════════════════════════════════════
    
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(function(input) {
        // Add step buttons for better UX
        const wrapper = document.createElement('div');
        wrapper.className = 'input-group';
        
        const step = parseFloat(input.step) || 1;
        
        // Create decrement button
        const decButton = document.createElement('button');
        decButton.type = 'button';
        decButton.className = 'btn btn-outline-secondary';
        decButton.innerHTML = '<i class="fas fa-minus"></i>';
        decButton.addEventListener('click', function() {
            const currentValue = parseFloat(input.value) || 0;
            const min = parseFloat(input.min);
            const newValue = currentValue - step;
            if (min === undefined || newValue >= min) {
                input.value = Math.round(newValue * 100) / 100;
                input.dispatchEvent(new Event('input'));
            }
        });
        
        // Create increment button
        const incButton = document.createElement('button');
        incButton.type = 'button';
        incButton.className = 'btn btn-outline-secondary';
        incButton.innerHTML = '<i class="fas fa-plus"></i>';
        incButton.addEventListener('click', function() {
            const currentValue = parseFloat(input.value) || 0;
            const max = parseFloat(input.max);
            const newValue = currentValue + step;
            if (max === undefined || newValue <= max) {
                input.value = Math.round(newValue * 100) / 100;
                input.dispatchEvent(new Event('input'));
            }
        });
        
        // Wrap input
        input.parentNode.insertBefore(wrapper, input);
        wrapper.appendChild(decButton);
        wrapper.appendChild(input);
        wrapper.appendChild(incButton);
        
        // Adjust input styling
        input.classList.remove('form-control');
        input.classList.add('form-control');
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Auto-save Form Data (Optional Feature)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    const FORM_STORAGE_KEY = 'learnsight_form_data';
    
    function saveFormData() {
        if (!predictionForm) return;
        
        const formData = new FormData(predictionForm);
        const data = {};
        
        formData.forEach(function(value, key) {
            data[key] = value;
        });
        
        localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(data));
    }
    
    function loadFormData() {
        const savedData = localStorage.getItem(FORM_STORAGE_KEY);
        if (!savedData || !predictionForm) return;
        
        try {
            const data = JSON.parse(savedData);
            
            Object.keys(data).forEach(function(key) {
                const input = predictionForm.querySelector(`[name="${key}"]`);
                if (input && !input.value) {
                    input.value = data[key];
                }
            });
        } catch (e) {
            console.error('Error loading saved form data:', e);
        }
    }
    
    // Auto-save on input change
    if (predictionForm) {
        const inputs = predictionForm.querySelectorAll('input, select');
        inputs.forEach(function(input) {
            input.addEventListener('change', saveFormData);
        });
        
        // Load saved data on page load
        loadFormData();
        
        // Clear saved data on successful submission
        predictionForm.addEventListener('submit', function() {
            localStorage.removeItem(FORM_STORAGE_KEY);
        });
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Chart Animations
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function animateCharts() {
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(function(chart) {
            // Trigger Plotly animation if available
            if (typeof Plotly !== 'undefined') {
                Plotly.animate(chart);
            }
        });
    }
    
    // Animate charts when they come into view
    if ('IntersectionObserver' in window) {
        const chartObserver = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    animateCharts();
                    chartObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        
        document.querySelectorAll('.js-plotly-plot').forEach(function(chart) {
            chartObserver.observe(chart);
        });
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Keyboard Shortcuts
    // ═══════════════════════════════════════════════════════════════════════════════
    
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (predictionForm && document.activeElement.closest('#predictionForm')) {
                e.preventDefault();
                predictionForm.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to close notifications
        if (e.key === 'Escape') {
            const notifications = document.querySelectorAll('.learnsight-notification');
            notifications.forEach(function(n) {
                n.remove();
            });
        }
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Print Functionality Enhancement
    // ═══════════════════════════════════════════════════════════════════════════════
    
    window.addEventListener('beforeprint', function() {
        // Ensure all charts are rendered before printing
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(function(chart) {
            if (typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(chart);
            }
        });
    });
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // API Helper Functions
    // ═══════════════════════════════════════════════════════════════════════════════
    
    window.LearnSightAPI = {
        /**
         * Make a prediction via API
         * @param {Object} data - Student data
         * @returns {Promise} - API response
         */
        predict: async function(data) {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return response.json();
        },
        
        /**
         * Get model information
         * @returns {Promise} - API response
         */
        getModels: async function() {
            const response = await fetch('/api/models');
            return response.json();
        }
    };
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // Console Welcome Message
    // ═══════════════════════════════════════════════════════════════════════════════
    
    console.log('%c🎓 LearnSight', 'font-size: 24px; font-weight: bold; color: #2C3E95;');
    console.log('%cStudent Academic Performance Prediction System', 'font-size: 14px; color: #4CAF50;');
    console.log('%cVersion: 1.0.0', 'font-size: 12px; color: #666;');
    console.log('%cEmpowering Educators with Explainable AI', 'font-size: 12px; color: #666;');
    
});

// ═══════════════════════════════════════════════════════════════════════════════
// Global Error Handler
// ═══════════════════════════════════════════════════════════════════════════════

window.addEventListener('error', function(e) {
    console.error('LearnSight Error:', e.error);
    
    // Show user-friendly error message
    if (typeof showNotification === 'function') {
        showNotification('An unexpected error occurred. Please try again.', 'error');
    }
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('LearnSight Unhandled Promise Rejection:', e.reason);
});
