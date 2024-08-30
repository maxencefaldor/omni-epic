// 'use client'
// import React, { useEffect, useState } from 'react';
// import io from 'socket.io-client';

// const VideoComponent = () => {
//   const [src, setSrc] = useState('');

//   useEffect(() => {
//     const socket = io('http://localhost:3004');  // Update with your Flask app's URL

//     socket.on('video_frame', data => {
//         console.log(data);
//       setSrc(`data:image/jpeg;base64,${data.data}`);
//     });

//     return () => socket.disconnect();
//   }, []);

//   return <img src={src} alt="Video Feed" />;
// };

// export default VideoComponent;
