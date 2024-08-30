import { SocketIdentifier } from "@/app/KeyIdentifier";
import Link from "next/link";

export function HomePage() {
  return (
    <div className="flex flex-col h-screen w-full">
      {/* <div className="flex w-full shrink-0 items-center px-4 border-b border-gray-200 dark:border-gray-800">
        <Link className="flex shrink-0 items-center space-x-2 text-lg font-semibold" href="/">
          omni_epic
        </Link>
      </div> */}
      <main className="flex flex-1 w-full flex-col items-center justify-center p-4">
        <div className="flex w-full max-w-7xl flex-col items-center justify-center gap-4">
          <div className="aspect-[16/9] w-full overflow-hidden rounded-lg shadow-lg">
            {/* <video className="w-full h-full object-cover rounded-md bg-gray-100 dark:bg-gray-800" controls> */}
            <img src="http://localhost:3005/video_feed" alt="Game Stream"  className="w-full h-full border-rad object-cover rounded-md bg-gray-100 dark:bg-gray-800" />
              Your browser does not support the video tag.
            {/* </video> */}
          
          </div>
          <SocketIdentifier></SocketIdentifier>
          {/* <div className="grid w-full gap-4">
            <h2 className="text-3xl font-bold">Try to beat the leaderboard</h2> */}
            {/* <h3 className="text-2xl font-medium">The keys that you can use are WASD</h3> */}
          {/* </div> */}
        </div>
      </main>
   
    </div>
  )
}