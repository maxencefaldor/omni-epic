"use client"
import React, { useEffect, useRef } from 'react';

export function CanvasVideoPlayer({ src }:any) {
  const canvasRef = useRef(null);
  const imageRef = useRef(new Image());

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const image = imageRef.current;

    const fetchStream = async () => {
      const response = await fetch(src);
      const reader = response.body.getReader();
      console.log(reader);
      let receivedLength = 0;
      let chunks = []; // Array of received binary chunks (comprises the body)
      while(true) {
        const {done, value} = await reader.read();
        if (done) {
          break;
        }
        chunks.push(value);
        receivedLength += value.length;

        const blob = new Blob(chunks, {type: "image/jpeg"});
        image.src = URL.createObjectURL(blob);
        
        // Clear the canvas and draw the new frame
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        chunks = []; // Clear the chunks for the next frame
      }
    };

    fetchStream();
  }, [src]);

  return <canvas ref={canvasRef} className="bg-black" width="640" height="480"></canvas>;
};
