// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// 滚动时导航栏效果和背景视频淡出
let lastScroll = 0;
const navbar = document.querySelector('.navbar');
const backgroundVideo = document.getElementById('background-video');
const header = document.querySelector('.header');
const highlightsSection = document.getElementById('highlights');

function updateVideoFade() {
    if (!backgroundVideo || !highlightsSection) return;
    
    const currentScroll = window.pageYOffset;
    const headerHeight = header.offsetHeight;
    const highlightsTop = highlightsSection.offsetTop;
    
    // 导航栏背景变化
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(0, 0, 0, 0.95)';
    } else {
        navbar.style.background = 'rgba(0, 0, 0, 0.8)';
    }
    
    // 背景视频淡出效果
    // 从 header 底部开始淡出，到 highlights section 顶部完全消失
    const fadeStart = headerHeight * 0.7; // 从 header 高度的 70% 开始淡出
    const fadeEnd = highlightsTop - 100; // 到 highlights section 前 100px 完全消失
    
    if (currentScroll > fadeStart) {
        const fadeProgress = Math.min((currentScroll - fadeStart) / (fadeEnd - fadeStart), 1);
        backgroundVideo.style.opacity = Math.max(0, 1 - fadeProgress);
        
        if (fadeProgress >= 1) {
            backgroundVideo.classList.add('fade-out');
        } else {
            backgroundVideo.classList.remove('fade-out');
        }
    } else {
        backgroundVideo.style.opacity = 1;
        backgroundVideo.classList.remove('fade-out');
    }
    
    lastScroll = currentScroll;
}

window.addEventListener('scroll', updateVideoFade);

// 页面加载和窗口大小改变时重新计算
window.addEventListener('resize', updateVideoFade);
window.addEventListener('load', updateVideoFade);

// 元素进入视口时的动画
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// 观察所有需要动画的元素
document.querySelectorAll('.video-container, .highlight-card, .bibtex-container').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
});

// Generalization section 视频自动播放
const generalizationSection = document.getElementById('generalization');
if (generalizationSection) {
    const generalizationVideos = generalizationSection.querySelectorAll('video');
    
    // 确保所有视频都是muted的
    generalizationVideos.forEach(video => {
        video.muted = true;
    });
    
    // 使用 Intersection Observer 来检测 section 是否进入视口
    const generalizationObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // 当 section 进入视口时，播放所有视频
                const videos = entry.target.querySelectorAll('video');
                videos.forEach(video => {
                    video.muted = true;
                    video.play().catch(err => {
                        console.log('Generalization video play error:', err);
                    });
                });
            } else {
                // 当 section 离开视口时，暂停所有视频
                const videos = entry.target.querySelectorAll('video');
                videos.forEach(video => {
                    video.pause();
                });
            }
        });
    }, {
        threshold: 0.1, // 当10%的section可见时触发
        rootMargin: '0px'
    });
    
    generalizationObserver.observe(generalizationSection);
}

// 背景视频轮播功能 - 一个播完再播下一个
function initVideoCarousel() {
    const videos = document.querySelectorAll('.bg-video');
    if (videos.length <= 1) return; // 如果只有一个或没有视频，不需要轮播
    
    let currentIndex = 0;
    
    // 预加载所有视频
    videos.forEach((video, index) => {
        if (index > 0) {
            video.load(); // 预加载视频
        }
        
        // 监听视频播放结束事件
        video.addEventListener('ended', function() {
            // 移除当前视频的 active 类
            videos[currentIndex].classList.remove('active');
            
            // 切换到下一个视频
            currentIndex = (currentIndex + 1) % videos.length;
            
            // 重置并播放下一个视频
            videos[currentIndex].currentTime = 0;
            videos[currentIndex].classList.add('active');
            videos[currentIndex].play().catch(err => {
                console.log('Video play error:', err);
            });
        });
    });
}

// 初始化视频轮播
initVideoCarousel();

// 交互式视频播放器功能
function initInteractiveVideoPlayer(container) {
    // 如果没有提供容器，查找当前激活的demo
    if (!container) {
        const activeDemo = document.querySelector('.demo-content.active');
        if (activeDemo) {
            container = activeDemo;
        } else {
            // 如果没有激活的demo，使用第一个demo
            container = document.querySelector('.demo-content');
        }
    }
    
    if (!container) return;
    
    const videoPlayer = container.querySelector('video');
    if (!videoPlayer) return;

    const videoProgress = container.querySelector('.video-progress');
    const videoTimeline = container.querySelector('.video-timeline');
    const videoTime = container.querySelector('.video-time');
    const sceneItems = container.querySelectorAll('.scene-item');
    const scenesList = container.querySelector('.scenes-list');
    const shotTextPanel = container.querySelector('.shot-text-panel');
    const shotTextHeader = container.querySelector('.shot-text-header');
    const shotTextDisplay = container.querySelector('#shotTextDisplay') || container.querySelector('.shot-text-content p');

    let videoDuration = 0;

    // 初始化第一个shot的文本
    const setInitialText = function() {
        // 重新查询以确保获取最新的元素
        const currentShotTextDisplay = container.querySelector('#shotTextDisplay') || container.querySelector('.shot-text-content p');
        const currentFirstActiveShot = container.querySelector('.scene-item.active') || container.querySelector('.scene-item');
        
        if (currentFirstActiveShot && currentShotTextDisplay) {
            const firstShotText = currentFirstActiveShot.getAttribute('data-text');
            if (firstShotText) {
                // 如果文本是空的或者是默认提示文本，则设置
                if (!currentShotTextDisplay.textContent || 
                    currentShotTextDisplay.textContent.includes('Click on a shot') ||
                    currentShotTextDisplay.textContent.trim() === '') {
                    currentShotTextDisplay.textContent = firstShotText;
                }
            }
        }
    };
    
    // 立即设置初始文本
    setInitialText();

    // 获取视频时长
    const metadataHandler = function() {
        videoDuration = videoPlayer.duration;
        updateTimeDisplay(0, videoDuration);
        // 确保初始shot文本已设置
        setInitialText();
    };
    // 移除旧的监听器（如果存在）
    videoPlayer.removeEventListener('loadedmetadata', metadataHandler);
    videoPlayer.addEventListener('loadedmetadata', metadataHandler);
    
    // 如果视频已经加载完成，立即执行
    if (videoPlayer.readyState >= 1) {
        metadataHandler();
    }

    // Shot点击跳转 - 使用事件委托避免重复绑定
    if (scenesList) {
        // 移除旧的监听器（如果存在）
        const clickHandler = function(e) {
            const sceneItem = e.target.closest('.scene-item');
            if (!sceneItem) return;
            
            // 确保sceneItem属于当前container
            if (!container.contains(sceneItem)) return;
            
            const jumpTime = parseFloat(sceneItem.getAttribute('data-time'));
            const shotText = sceneItem.getAttribute('data-text');
            
            if (isNaN(jumpTime)) return;
            
            videoPlayer.currentTime = jumpTime;
            videoPlayer.play();

            // 更新active shot - 重新查询以确保获取最新的元素
            const currentSceneItems = container.querySelectorAll('.scene-item');
            currentSceneItems.forEach(scene => {
                scene.classList.remove('active');
            });
            sceneItem.classList.add('active');

            // 更新文本
            if (shotText) {
                const currentShotTextDisplay = container.querySelector('#shotTextDisplay') || container.querySelector('.shot-text-content p');
                if (currentShotTextDisplay) {
                    currentShotTextDisplay.textContent = shotText;
                    // 自动展开文本面板
                    const currentShotTextPanel = container.querySelector('.shot-text-panel');
                    if (currentShotTextPanel && !currentShotTextPanel.classList.contains('expanded')) {
                        currentShotTextPanel.classList.add('expanded');
                    }
                }
            }

            // 滚动到可见区域
            const itemRect = sceneItem.getBoundingClientRect();
            const listRect = scenesList.getBoundingClientRect();
            
            if (itemRect.left < listRect.left || itemRect.right > listRect.right) {
                const scrollLeft = sceneItem.offsetLeft - (listRect.width - itemRect.width) / 2;
                scenesList.scrollTo({
                    left: scrollLeft,
                    behavior: 'smooth'
                });
            }
        };
        
        // 存储handler到元素上以便后续移除
        if (scenesList._clickHandler) {
            scenesList.removeEventListener('click', scenesList._clickHandler);
        }
        scenesList._clickHandler = clickHandler;
        scenesList.addEventListener('click', clickHandler);
    }

    // 更新进度条和时间
    const timeupdateHandler = function() {
        const currentTime = videoPlayer.currentTime;
        const progressPercent = (currentTime / videoDuration) * 100;
        
        if (videoProgress) {
            videoProgress.style.width = `${progressPercent}%`;
        }
        updateTimeDisplay(currentTime, videoDuration);
        updateActiveScene(currentTime);
    };
    // 移除旧的监听器（如果存在）
    videoPlayer.removeEventListener('timeupdate', timeupdateHandler);
    videoPlayer.addEventListener('timeupdate', timeupdateHandler);

    // 时间轴点击跳转
    if (videoTimeline) {
        const timelineClickHandler = function(e) {
            const rect = this.getBoundingClientRect();
            const position = (e.clientX - rect.left) / rect.width;
            const seekTime = position * videoDuration;
            
            videoPlayer.currentTime = seekTime;
            videoPlayer.play();
        };
        // 移除旧的监听器（如果存在）
        if (videoTimeline._clickHandler) {
            videoTimeline.removeEventListener('click', videoTimeline._clickHandler);
        }
        videoTimeline._clickHandler = timelineClickHandler;
        videoTimeline.addEventListener('click', timelineClickHandler);
    }

    // 格式化时间
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // 更新时间显示
    function updateTimeDisplay(currentTime, duration) {
        if (videoTime) {
            videoTime.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
        }
    }

    // 更新active shot
    function updateActiveScene(currentTime) {
        // 重新查询以确保获取最新的元素
        const currentSceneItems = container.querySelectorAll('.scene-item');
        const currentShotTextDisplay = container.querySelector('#shotTextDisplay') || container.querySelector('.shot-text-content p');
        const currentScenesList = container.querySelector('.scenes-list');
        
        if (currentSceneItems.length === 0) return;
        
        let activeSceneFound = false;
        
        currentSceneItems.forEach((item, index) => {
            const itemTime = parseFloat(item.getAttribute('data-time'));
            const nextItemTime = index < currentSceneItems.length - 1 
                ? parseFloat(currentSceneItems[index + 1].getAttribute('data-time')) 
                : videoDuration;
            
            if (currentTime >= itemTime && currentTime < nextItemTime) {
                if (!item.classList.contains('active')) {
                    // 更新active状态
                    currentSceneItems.forEach(scene => {
                        scene.classList.remove('active');
                    });
                    item.classList.add('active');

                    // 更新文本
                    const shotText = item.getAttribute('data-text');
                    if (shotText && currentShotTextDisplay) {
                        currentShotTextDisplay.textContent = shotText;
                    }

                    // 自动滚动到可见区域
                    if (currentScenesList) {
                        const itemRect = item.getBoundingClientRect();
                        const listRect = currentScenesList.getBoundingClientRect();
                        
                        if (itemRect.left < listRect.left || itemRect.right > listRect.right) {
                            const scrollLeft = item.offsetLeft - (listRect.width - itemRect.width) / 2;
                            currentScenesList.scrollTo({
                                left: scrollLeft,
                                behavior: 'smooth'
                            });
                        }
                    }
                }
                activeSceneFound = true;
            }
        });
        
        // 默认第一个shot
        if (!activeSceneFound && currentSceneItems.length > 0) {
            const firstItem = currentSceneItems[0];
            if (!firstItem.classList.contains('active')) {
                currentSceneItems.forEach(scene => {
                    scene.classList.remove('active');
                });
                firstItem.classList.add('active');
                
                // 更新第一个shot的文本
                const firstShotText = firstItem.getAttribute('data-text');
                if (firstShotText && currentShotTextDisplay) {
                    currentShotTextDisplay.textContent = firstShotText;
                }
            }
        }
    }

    // 文本面板展开/折叠
    if (shotTextHeader && shotTextPanel) {
        const headerClickHandler = function() {
            shotTextPanel.classList.toggle('expanded');
        };
        // 移除旧的监听器（如果存在）
        if (shotTextHeader._clickHandler) {
            shotTextHeader.removeEventListener('click', shotTextHeader._clickHandler);
        }
        shotTextHeader._clickHandler = headerClickHandler;
        shotTextHeader.addEventListener('click', headerClickHandler);
    }
}

// 初始化交互式视频播放器
// 确保在DOM完全加载后再初始化第一个demo
function initFirstDemo() {
    const firstDemo = document.querySelector('.demo-content.active') || document.querySelector('.demo-content');
    if (firstDemo) {
        // 延迟一点确保DOM完全准备好
        setTimeout(() => {
            initInteractiveVideoPlayer(firstDemo);
        }, 100);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        initFirstDemo();
    });
} else {
    // DOM已经加载，但可能需要等待一下
    if (document.readyState === 'complete') {
        initFirstDemo();
    } else {
        window.addEventListener('load', initFirstDemo);
    }
}

// Demo翻页功能
function initDemoTabs() {
    const demoTabs = document.querySelectorAll('.demo-tab');
    const demoContents = document.querySelectorAll('.demo-content');
    if (demoTabs.length === 0 || demoContents.length === 0) return;
    
    let autoSwitchTimer = null;
    let isUserInteracting = false;
    let currentDemoIndex = 0;
    const AUTO_SWITCH_INTERVAL = 60000; // 30秒自动切换

    // 切换demo函数
    function switchDemo(index) {
        if (index < 0 || index >= demoContents.length) return;
        
        currentDemoIndex = index;
        
        // 移除所有active状态
        demoTabs.forEach(tab => tab.classList.remove('active'));
        demoContents.forEach(content => content.classList.remove('active'));

        // 激活选中的demo
        if (demoTabs[index]) {
            demoTabs[index].classList.add('active');
        }
        if (demoContents[index]) {
            demoContents[index].classList.add('active');
        }

        // 重新初始化当前demo的视频播放器（如果有）
        const currentContent = demoContents[index];
        if (currentContent) {
            const videoPlayer = currentContent.querySelector('video');
            if (videoPlayer && typeof initInteractiveVideoPlayer === 'function') {
                // 延迟初始化以确保DOM已更新
                setTimeout(() => {
                    initInteractiveVideoPlayer(currentContent);
                }, 100);
            }
        }
    }

    // Tab点击事件
    demoTabs.forEach((tab, index) => {
        tab.addEventListener('click', function() {
            isUserInteracting = true;
            clearInterval(autoSwitchTimer);
            switchDemo(index);
            
            // 用户交互后，暂停自动切换一段时间
            setTimeout(() => {
                isUserInteracting = false;
                startAutoSwitch();
            }, 30000); // 30秒后恢复自动切换
        });
    });

    // 左右键切换按钮
    const demoNavLeft = document.getElementById('demoNavLeft');
    const demoNavRight = document.getElementById('demoNavRight');
    
    function navigateDemo(direction) {
        isUserInteracting = true;
        clearInterval(autoSwitchTimer);
        if (direction === 'prev') {
            const prevIndex = (currentDemoIndex - 1 + demoContents.length) % demoContents.length;
            switchDemo(prevIndex);
        } else if (direction === 'next') {
            const nextIndex = (currentDemoIndex + 1) % demoContents.length;
            switchDemo(nextIndex);
        }
        setTimeout(() => {
            isUserInteracting = false;
            startAutoSwitch();
        }, 30000);
    }
    
    if (demoNavLeft) {
        demoNavLeft.addEventListener('click', () => navigateDemo('prev'));
    }
    
    if (demoNavRight) {
        demoNavRight.addEventListener('click', () => navigateDemo('next'));
    }

    // 键盘左右键切换
    document.addEventListener('keydown', function(e) {
        // 只在视频容器可见时响应键盘事件
        const videoContainer = document.querySelector('.video-container');
        if (!videoContainer) return;
        
        const rect = videoContainer.getBoundingClientRect();
        const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
        
        if (!isVisible) return;
        
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            navigateDemo('prev');
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            navigateDemo('next');
        }
    });

    // 自动切换函数
    function startAutoSwitch() {
        if (isUserInteracting) return;
        
        clearInterval(autoSwitchTimer);
        autoSwitchTimer = setInterval(() => {
            if (!isUserInteracting) {
                currentDemoIndex = (currentDemoIndex + 1) % demoContents.length;
                switchDemo(currentDemoIndex);
            }
        }, AUTO_SWITCH_INTERVAL);
    }

    // 鼠标进入时暂停自动切换
    const videoContainer = document.querySelector('.video-container');
    if (videoContainer) {
        videoContainer.addEventListener('mouseenter', function() {
            isUserInteracting = true;
            clearInterval(autoSwitchTimer);
        });

        // 鼠标离开后恢复自动切换（延迟）
        videoContainer.addEventListener('mouseleave', function() {
            setTimeout(() => {
                isUserInteracting = false;
                startAutoSwitch();
            }, 2000);
        });
    }

    // 初始化：开始自动切换
    startAutoSwitch();
}

// 初始化Demo翻页
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDemoTabs);
} else {
    initDemoTabs();
}

// Comparison翻页功能
function initComparisonTabs() {
    const comparisonTabs = document.querySelectorAll('.comparison-tab');
    const comparisonPages = document.querySelectorAll('.comparison-page');
    if (comparisonTabs.length === 0 || comparisonPages.length === 0) return;
    
    let autoSwitchTimer = null;
    let isUserInteracting = false;
    let currentPageIndex = 0;
    const AUTO_SWITCH_INTERVAL = 60000; // 60秒自动切换

    // 切换页面函数
    function switchPage(index) {
        if (index < 0 || index >= comparisonPages.length) return;
        
        currentPageIndex = index;
        
        // 暂停所有页面的所有video
        comparisonPages.forEach(page => {
            const videos = page.querySelectorAll('video');
            videos.forEach(video => {
                video.pause();
                video.currentTime = 0; // 重置到开头
            });
            page.classList.remove('active');
        });
        
        // 移除所有tab的active状态
        comparisonTabs.forEach(tab => tab.classList.remove('active'));

        // 激活选中的页面
        if (comparisonTabs[index]) {
            comparisonTabs[index].classList.add('active');
        }
        if (comparisonPages[index]) {
            comparisonPages[index].classList.add('active');
            
            // 播放当前页面的所有video
            const currentVideos = comparisonPages[index].querySelectorAll('video');
            currentVideos.forEach(video => {
                // 确保video是muted的（特别是Sora视频）
                video.muted = true;
                video.play().catch(err => {
                    console.log('Video play error:', err);
                });
            });
        }
    }

    // Tab点击事件
    comparisonTabs.forEach((tab, index) => {
        tab.addEventListener('click', function() {
            isUserInteracting = true;
            clearInterval(autoSwitchTimer);
            switchPage(index);
            
            // 用户交互后，暂停自动切换一段时间
            setTimeout(() => {
                isUserInteracting = false;
                startAutoSwitch();
            }, 30000); // 30秒后恢复自动切换
        });
    });

    // 左右键切换按钮
    const comparisonNavLeft = document.getElementById('comparisonNavLeft');
    const comparisonNavRight = document.getElementById('comparisonNavRight');
    
    function navigatePage(direction) {
        isUserInteracting = true;
        clearInterval(autoSwitchTimer);
        if (direction === 'prev') {
            const prevIndex = (currentPageIndex - 1 + comparisonPages.length) % comparisonPages.length;
            switchPage(prevIndex);
        } else if (direction === 'next') {
            const nextIndex = (currentPageIndex + 1) % comparisonPages.length;
            switchPage(nextIndex);
        }
        setTimeout(() => {
            isUserInteracting = false;
            startAutoSwitch();
        }, 30000);
    }
    
    if (comparisonNavLeft) {
        comparisonNavLeft.addEventListener('click', () => navigatePage('prev'));
    }
    
    if (comparisonNavRight) {
        comparisonNavRight.addEventListener('click', () => navigatePage('next'));
    }

    // 键盘导航
    document.addEventListener('keydown', function(e) {
        const comparisonSection = document.getElementById('comparison');
        if (!comparisonSection) return;
        
        const rect = comparisonSection.getBoundingClientRect();
        const isInView = rect.top < window.innerHeight && rect.bottom > 0;
        
        if (isInView && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
            e.preventDefault();
            if (e.key === 'ArrowLeft') {
                navigatePage('prev');
            } else {
                navigatePage('next');
            }
        }
    });

    // 自动切换功能
    function startAutoSwitch() {
        clearInterval(autoSwitchTimer);
        autoSwitchTimer = setInterval(() => {
            if (!isUserInteracting) {
                currentPageIndex = (currentPageIndex + 1) % comparisonPages.length;
                switchPage(currentPageIndex);
            }
        }, AUTO_SWITCH_INTERVAL);
    }

    // 鼠标悬停时暂停自动切换
    const comparisonContainer = document.querySelector('#comparison .video-container');
    if (comparisonContainer) {
        comparisonContainer.addEventListener('mouseenter', () => {
            clearInterval(autoSwitchTimer);
        });

        // 鼠标离开后恢复自动切换（延迟）
        comparisonContainer.addEventListener('mouseleave', () => {
            setTimeout(() => {
                if (!isUserInteracting) {
                    startAutoSwitch();
                }
            }, 2000);
        });
    }

    // 开始自动切换
    startAutoSwitch();
    
    // 初始化第一个页面的video播放
    const firstPage = comparisonPages[0];
    if (firstPage && firstPage.classList.contains('active')) {
        const firstVideos = firstPage.querySelectorAll('video');
        firstVideos.forEach(video => {
            // 确保video是muted的（特别是Sora视频）
            video.muted = true;
            video.play().catch(err => {
                console.log('Video play error:', err);
            });
        });
    }
    
    // 确保所有comparison video都是muted的（防止Sora视频有声音）
    const allComparisonVideos = document.querySelectorAll('#comparison video');
    allComparisonVideos.forEach(video => {
        video.muted = true;
    });
}

// 初始化Comparison翻页功能
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initComparisonTabs);
} else {
    initComparisonTabs();
}

// MR2V翻页功能
function initMR2VTabs() {
    const mr2vTabs = document.querySelectorAll('.mr2v-tab');
    const mr2vPages = document.querySelectorAll('.mr2v-page');
    if (mr2vTabs.length === 0 || mr2vPages.length === 0) return;
    
    let autoSwitchTimer = null;
    let isUserInteracting = false;
    let currentPageIndex = 0;
    const AUTO_SWITCH_INTERVAL = 60000; // 60秒自动切换

    // 切换页面函数
    function switchPage(index) {
        if (index < 0 || index >= mr2vPages.length) return;
        
        currentPageIndex = index;
        
        // 暂停所有页面的所有video
        mr2vPages.forEach(page => {
            const videos = page.querySelectorAll('video');
            videos.forEach(video => {
                video.pause();
                video.currentTime = 0; // 重置到开头
            });
            page.classList.remove('active');
        });
        
        // 移除所有tab的active状态
        mr2vTabs.forEach(tab => tab.classList.remove('active'));

        // 激活选中的页面
        if (mr2vTabs[index]) {
            mr2vTabs[index].classList.add('active');
        }
        if (mr2vPages[index]) {
            mr2vPages[index].classList.add('active');
            
            // 播放当前页面的所有video
            const currentVideos = mr2vPages[index].querySelectorAll('video');
            currentVideos.forEach(video => {
                // 确保video是muted的
                video.muted = true;
                video.play().catch(err => {
                    console.log('Video play error:', err);
                });
            });
        }
    }

    // Tab点击事件
    mr2vTabs.forEach((tab, index) => {
        tab.addEventListener('click', function() {
            isUserInteracting = true;
            clearInterval(autoSwitchTimer);
            switchPage(index);
            
            // 用户交互后，暂停自动切换一段时间
            setTimeout(() => {
                isUserInteracting = false;
                startAutoSwitch();
            }, 30000); // 30秒后恢复自动切换
        });
    });

    // 左右键切换按钮
    const mr2vNavLeft = document.getElementById('mr2vNavLeft');
    const mr2vNavRight = document.getElementById('mr2vNavRight');
    
    function navigatePage(direction) {
        isUserInteracting = true;
        clearInterval(autoSwitchTimer);
        if (direction === 'prev') {
            const prevIndex = (currentPageIndex - 1 + mr2vPages.length) % mr2vPages.length;
            switchPage(prevIndex);
        } else if (direction === 'next') {
            const nextIndex = (currentPageIndex + 1) % mr2vPages.length;
            switchPage(nextIndex);
        }
        setTimeout(() => {
            isUserInteracting = false;
            startAutoSwitch();
        }, 30000);
    }
    
    if (mr2vNavLeft) {
        mr2vNavLeft.addEventListener('click', () => navigatePage('prev'));
    }
    
    if (mr2vNavRight) {
        mr2vNavRight.addEventListener('click', () => navigatePage('next'));
    }

    // 键盘导航
    document.addEventListener('keydown', function(e) {
        const mr2vSection = document.getElementById('mr2v');
        if (!mr2vSection) return;
        
        const rect = mr2vSection.getBoundingClientRect();
        const isInView = rect.top < window.innerHeight && rect.bottom > 0;
        
        if (isInView && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
            // 检查是否在comparison section，如果是则不处理
            const comparisonSection = document.getElementById('comparison');
            if (comparisonSection) {
                const comparisonRect = comparisonSection.getBoundingClientRect();
                const comparisonInView = comparisonRect.top < window.innerHeight && comparisonRect.bottom > 0;
                if (comparisonInView) return; // 如果comparison也在视图中，优先处理comparison
            }
            
            e.preventDefault();
            if (e.key === 'ArrowLeft') {
                navigatePage('prev');
            } else {
                navigatePage('next');
            }
        }
    });

    // 自动切换功能
    function startAutoSwitch() {
        clearInterval(autoSwitchTimer);
        autoSwitchTimer = setInterval(() => {
            if (!isUserInteracting) {
                currentPageIndex = (currentPageIndex + 1) % mr2vPages.length;
                switchPage(currentPageIndex);
            }
        }, AUTO_SWITCH_INTERVAL);
    }

    // 鼠标悬停时暂停自动切换
    const mr2vContainer = document.querySelector('#mr2v .video-container');
    if (mr2vContainer) {
        mr2vContainer.addEventListener('mouseenter', () => {
            clearInterval(autoSwitchTimer);
        });

        // 鼠标离开后恢复自动切换（延迟）
        mr2vContainer.addEventListener('mouseleave', () => {
            setTimeout(() => {
                if (!isUserInteracting) {
                    startAutoSwitch();
                }
            }, 2000);
        });
    }

    // 开始自动切换
    startAutoSwitch();
    
    // 初始化第一个页面的video播放
    const firstPage = mr2vPages[0];
    if (firstPage && firstPage.classList.contains('active')) {
        const firstVideos = firstPage.querySelectorAll('video');
        firstVideos.forEach(video => {
            // 确保video是muted的
            video.muted = true;
            video.play().catch(err => {
                console.log('Video play error:', err);
            });
        });
    }
    
    // 确保所有MR2V video都是muted的
    const allMR2VVideos = document.querySelectorAll('#mr2v video');
    allMR2VVideos.forEach(video => {
        video.muted = true;
    });
}

// 初始化MR2V翻页功能
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMR2VTabs);
} else {
    initMR2VTabs();
}

// 复制 BibTeX 功能
function copyBibtex() {
    const bibtexContent = document.getElementById('bibtex-content').textContent;
    const btn = document.querySelector('.copy-btn');
    
    // 使用现代 Clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(bibtexContent).then(() => {
            showCopySuccess(btn);
        }).catch(err => {
            // 如果 Clipboard API 失败，使用备用方法
            fallbackCopy(bibtexContent, btn);
        });
    } else {
        // 使用备用方法
        fallbackCopy(bibtexContent, btn);
    }
}

// 备用复制方法
function fallbackCopy(text, btn) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-999999px';
    textarea.style.top = '-999999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    
    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showCopySuccess(btn);
        } else {
            alert('Failed to copy. Please select and copy manually.');
        }
    } catch (err) {
        console.error('Fallback copy failed: ', err);
        alert('Failed to copy. Please select and copy manually.');
    } finally {
        document.body.removeChild(textarea);
    }
}

// 显示复制成功提示
function showCopySuccess(btn) {
    const originalText = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.background = 'rgba(76, 175, 80, 0.3)';
    btn.style.borderColor = 'rgba(76, 175, 80, 0.5)';
    setTimeout(() => {
        btn.textContent = originalText;
        btn.style.background = 'rgba(255, 255, 255, 0.1)';
        btn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
    }, 2000);
}
