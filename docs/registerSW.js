if('serviceWorker' in navigator) {window.addEventListener('load', () => {navigator.serviceWorker.register('/track-n-take/sw.js', { scope: '/track-n-take/' })})}