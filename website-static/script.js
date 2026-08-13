// Scroll Progress Bar
window.addEventListener('scroll', () => {
    const scrollProgress = document.getElementById('scroll-progress');
    const scrolled = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
    scrollProgress.style.width = `${scrolled}%`;
});

// Intersection Observer for Animate-on-Scroll
document.addEventListener('DOMContentLoaded', () => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.15 });

    const animatedElements = document.querySelectorAll('.animate-up');
    animatedElements.forEach(el => observer.observe(el));

    setTimeout(() => {
        animatedElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight) {
                el.classList.add('visible');
            }
        });
    }, 100);

    // ── tsParticles: Minimal Hero Background ──
    if (typeof tsParticles !== 'undefined') {
        tsParticles.load("hero-particles", {
            fullScreen: { enable: false },
            fpsLimit: 60,
            particles: {
                number: {
                    value: 80,
                    density: { enable: true, area: 800 }
                },
                color: { value: ["#6b21a8", "#9333ea", "#a855f7"] },
                opacity: {
                    value: { min: 0.15, max: 0.5 },
                    animation: {
                        enable: true,
                        speed: 0.5,
                        minimumValue: 0.1,
                        sync: false
                    }
                },
                size: {
                    value: { min: 1.5, max: 4 }
                },
                move: {
                    enable: true,
                    speed: 0.4,
                    direction: "none",
                    outModes: { default: "out" },
                    random: true,
                    straight: false
                },
                links: {
                    enable: true,
                    color: "#6b21a8",
                    opacity: 0.18,
                    distance: 150,
                    width: 1
                },
                shape: { type: "circle" }
            },
            interactivity: {
                events: {
                    onHover: {
                        enable: true,
                        mode: "grab"
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 160,
                        links: { opacity: 0.35 }
                    }
                }
            },
            detectRetina: true
        });
    }
});
