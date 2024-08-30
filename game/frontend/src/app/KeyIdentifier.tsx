"use client"
import { Connected } from '@/components/ui/connected/connected';
import { NotConnected } from '@/components/ui/connected/notconnected';
import React, { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { Button } from "@/components/ui/button"
import { ToastAction } from "@/components/ui/toast"
import { useToast } from "@/components/ui/use-toast"
import { ToastDemo } from '@/components/ui/connected/toast';
import { EnvDescriptionEvent } from '../../types/socket_types';
import { ResetAlertDialog } from '@/components/ui/next_level/button_level';
export function SocketIdentifier() {
  const { toast } = useToast()
  const [connected, setConnected] = useState(false);
  const [levelFinished, setLevelFinishedToast] = useState(false); // State to track if the next level toast has been shown
  const [newSocket, setSocket] = useState<Socket | null>(null);
const [nextLevel, setNextLevelToastShown] = useState(false);
  // let socket: Socket;
  // useEffect(() => {
  //   // This side effect reacts to the change in level completion status.
  //   if (nextLevel) {
  //     // Show the toast for level completion.
  //     toast({
  //       title: "You've finished the first level!",
  //       duration: 900000, // 15 minutes or any desired duration
  //       description: 'You have reached the next level. Prepare for new challenges!',
  //       action: (
  //         <ToastAction altText="Dismiss">Dismiss</ToastAction>
  //       ),
  //     });
  //   }
  // }, [nextLevel, toast]);
  useEffect(() => {
    // This side effect reacts to the change in level completion status.
    if (levelFinished) {
      // Show the toast for level completion.
      toast({
        title: "You've finished the first level!",
        duration:3000, // 15 minutes or any desired duration
        description: 'You have reached the next level. Prepare for new challenges!',
        action: (
          <ToastAction altText="Dismiss">Dismiss</ToastAction>
        ),
      });
    }
  }, [levelFinished, toast]);
  useEffect(() => {

    const socket = io(process.env.NEXT_PUBLIC_API_URL!);
    setSocket(socket);
    
    // socket = io(process.env.NEXT_PUBLIC_API_URL!, {
    //   // Add any options here
    // });
    console.log(process.env.NEXT_PUBLIC_API_URL!);
    // socket.on('connect', () => {
    //   console.log('Connected to Socket.IO server');
    //   setConnected(true);
    //   // Show the toast notification on connect
    //   toast({
    //     title: 'Socket Connection Established',
    //     description: 'You are now connected to the server.',
    //     action: (
    //       <ToastAction altText="See connection details">Details</ToastAction>
    //     ),
    //   });
    // });
    socket.on('connect', () => {
      console.log('Connected to Socket.IO server');
      setConnected(true);
    });
    socket.on('disconnect', () => {
      console.log('Disconnected from Socket.IO server');
      setConnected(false);
    });
    socket.on('env_description', (data: EnvDescriptionEvent) => {
      toast({
        className: 'text-4xl ',
        title: 'Instructions for the current level:',
        description: <div>{data.description.split('\n').map(
          (line, index) => (
            <div key={index}>{line}<br/></div>
          )
        )}</div>,
        duration: 40000,
        action: (
          <ToastAction altText="View details" >Close</ToastAction>
        ),
      });
    });
    socket.on('reset_message',()=>{
      setLevelFinishedToast(false);
    })
    socket.on('level_complete', () => {
      console.log('Level completed')
      // Show a toast notification about starting the next level
      // console.log(levelFinished)
      // if (!levelFinished) {
      //   toast({
      //     title: 'Next Level!',
      //     duration:900000,
      //     description: 'You have reached the next level. Prepare for new challenges!',
      //     action: (
      //       <ToastAction altText="View level details">Level Details</ToastAction>
      //     ),
      //   });
      //   // Mark the toast as shown
        setLevelFinishedToast(true);
      // }


 

    });

    socket.on('next_level', () => {
      // console.log('Next level message received')
      // Show a toast notification about starting the next level
      // console.log(levelFinished)
      // if (!levelFinished) {
      //   toast({
      //     title: 'Next Level!',
      //     duration:900000,
      //     description: 'You have reached the next level. Prepare for new challenges!',
      //     action: (
      //       <ToastAction altText="View level details">Level Details</ToastAction>
      //     ),
      //   });
      //   // Mark the toast as shown
        setNextLevelToastShown(true);
      // }


 

    });




    const handleKeyDown = (event: KeyboardEvent) => {
      let action;
      switch (event.key) {
        case 'w':
        case 'ArrowUp':
          action = 1; // Go forward
          break;

        case 's':
        case 'ArrowDown':
          action = 2; // Go backward
          break;

        case 'a':
        case 'ArrowLeft':
          action = 3; // Rotate counterclockwise
          break;

        case 'd':
        case 'ArrowRight':
          action = 4; // Rotate clockwise
          break;

        case 'm':
        case 'Spacebar':
          action = 5; // Jump
          break;

        default:
          return;
      }
      if (connected && action !== undefined) {
        socket.emit('action', { action });
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('level_complete');
      socket.off('env_description'); // Clean up the listener
      socket.close();
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [connected, toast]); // Include 'toast' in the dependency array to ensure it's captured by useEffect
  const handleNextLevel = () => {
    newSocket?.emit('next_level');
  };

  const handleReset = () => {
    // setLevelFinishedToast(false);
    newSocket?.emit('reset');
 
  };
  return (
    <div className="flex flex-col items-stretch space-y-2">
      {/* <ToastDemo></ToastDemo> */}
      <div className="min-w-full">
        {connected ? <Connected></Connected> : <NotConnected></NotConnected>}
      </div>
      <Button onClick={handleNextLevel} className="w-full" >Next Level</Button>

      <div className="min-w-full">
        <ResetAlertDialog onConfirm={handleReset}></ResetAlertDialog>
      </div>

      {/* <button onClick={handleReset}>Reset Level</button> */}
      {/* {levelFinished ? <div>Level completed</div> : <div>Level not completed</div>} */}
    </div>
  );
}