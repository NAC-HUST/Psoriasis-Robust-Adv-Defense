// Smooth scroll behavior for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Navigation active state
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');

function updateActiveNav() {
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (scrollY >= sectionTop - 200) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === currentSection) {
            link.classList.add('active');
        }
    });
}

window.addEventListener('scroll', updateActiveNav);

// Sticky navbar shadow
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.05)';
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all cards and elements for animation
document.querySelectorAll('.overview-card, .tech-item, .step-card, .publication-card').forEach(el => {
    observer.observe(el);
});

// Code block styling with syntax highlighting
document.querySelectorAll('pre code').forEach(block => {
    // Apply basic syntax highlighting
    let code = block.innerHTML;
    
    // Keywords
    code = code.replace(/\b(uv|run|main|py|preprocess|download-models|train|attack|backbone|datadir|epochs|batch-size)\b/g, '<span class="keyword">$1</span>');
    
    // Flags
    code = code.replace(/--([a-z-]+)/g, '<span class="flag">--$1</span>');
    
    block.innerHTML = code;
});

// Copy to clipboard functionality
document.querySelectorAll('pre').forEach(pre => {
    const button = document.createElement('button');
    button.className = 'copy-btn';
    button.textContent = '📋 复制';
    button.style.cssText = `
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        padding: 0.5rem 1rem;
        background: #4a7fd4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 0.8rem;
        opacity: 0;
        transition: all 0.3s ease;
    `;
    
    pre.style.position = 'relative';
    pre.appendChild(button);
    
    pre.addEventListener('mouseenter', () => {
        button.style.opacity = '1';
    });
    
    pre.addEventListener('mouseleave', () => {
        button.style.opacity = '0';
    });
    
    button.addEventListener('click', async () => {
        const code = pre.querySelector('code').innerText;
        try {
            await navigator.clipboard.writeText(code);
            button.textContent = '✓ 已复制';
            setTimeout(() => {
                button.textContent = '📋 复制';
            }, 2000);
        } catch (err) {
            console.error('复制失败:', err);
        }
    });
});

// Dynamic counter animation
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// Observe stat elements
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
            const statValue = entry.target.querySelector('.stat-value');
            if (statValue) {
                const text = statValue.textContent;
                const number = parseInt(text);
                if (!isNaN(number)) {
                    animateCounter(statValue, number);
                }
            }
            entry.target.classList.add('animated');
            statsObserver.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll('.stat').forEach(stat => {
    statsObserver.observe(stat);
});

// Mobile menu toggle (if navigation becomes collapsible)
const navMenu = document.querySelector('.nav-menu');
const navbarContainer = document.querySelector('.navbar-container');

// Add mobile responsiveness
function handleResponsive() {
    if (window.innerWidth < 768) {
        // Mobile layout logic
    }
}

window.addEventListener('resize', handleResponsive);
handleResponsive();

// Scroll animation: reveal elements from bottom
const revealElements = document.querySelectorAll('.overview-card, .feature-item, .tech-item, .step-card');

revealElements.forEach((el, index) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.transition = 'all 0.6s ease-out';
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, index * 50);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    observer.observe(el);
});

// Add CSS for highlighted syntax
const style = document.createElement('style');
style.textContent = `
    .keyword {
        color: #10b981;
        font-weight: 500;
    }
    
    .flag {
        color: #f59e0b;
    }
    
    .copy-btn:hover {
        background: #1e5fb3;
        transform: translateY(-2px);
    }
    
    .nav-menu a.active {
        color: #ff6b5b;
        background: rgba(255, 107, 91, 0.1);
    }
`;
document.head.appendChild(style);

// Parallax effect for hero section
const hero = document.querySelector('.hero');
window.addEventListener('scroll', () => {
    if (hero) {
        const scrolled = window.scrollY;
        const blobs = hero.querySelectorAll('.gradient-blob');
        blobs.forEach((blob, index) => {
            const speed = (index + 1) * 0.5;
            blob.style.transform = `translateY(${scrolled * speed}px)`;
        });
    }
});

// Fade in on page load
window.addEventListener('load', () => {
    document.body.style.opacity = '1';
});

document.body.style.opacity = '0';
document.body.style.transition = 'opacity 0.5s ease-in';
