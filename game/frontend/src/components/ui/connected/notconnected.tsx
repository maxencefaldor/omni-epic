import { LinkBreak2Icon} from "@radix-ui/react-icons"

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"

export function NotConnected() {
  return (
    <Alert>
      <LinkBreak2Icon className="h-4 w-4" />
      <AlertTitle>   <div style={{color: 'red'}}>Not connected</div></AlertTitle>
      {/* <AlertDescription>
      <div style={{color: 'green'}}>Connected</div>
      </AlertDescription> */}

    </Alert>
  )
}
