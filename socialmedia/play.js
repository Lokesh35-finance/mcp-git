<!DOCTYPE html>
<html>
<head>
  <title>Play Video with JS</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background: #f0f2f5; }
    video { width: 70%; margin-top: 20px; border: 2px solid #333; border-radius: 10px; }
    button { margin: 10px; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; background: #1877f2; color: #fff; }
  </style>
</head>
<body>
  <h2>Video Player (JavaScript Example)</h2>

  <video id="myVideo" controls>
    <source src="sample.mp4" type="video/mp4">
    Your browser does not support HTML5 video.
  </video>
  <br>

  <button onclick="playVideo()">Play</button>
  <button onclick="pauseVideo()">Pause</button>
  <button onclick="stopVideo()">Stop</button>
  <button onclick="toggleMute()">Mute/Unmute</button>

  <script>
    const video = document.getElementById("myVideo");

    function playVideo() {
      video.play();
    }

    function pauseVideo() {
      video.pause();
    }

    function stopVideo() {
      video.pause();
      video.currentTime = 0; // Reset to start
    }

    function toggleMute() {
      video.muted = !video.muted;
    }
  </script>
</body>
</html>
