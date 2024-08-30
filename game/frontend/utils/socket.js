import io from 'socket.io-client';
let socket;

export const initSocket = () => {
  // Ensures a single socket connection is maintained
  if (!socket) {
    socket = io(process.env.NEXT_PUBLIC_API_URL, {
      // Add any options here
    });
    console.log('Connecting to Socket.IO server...');
  }
  return socket;
};
