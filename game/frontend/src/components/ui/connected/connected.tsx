import { LinkNone2Icon} from "@radix-ui/react-icons"

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"

export function Connected() {
  return (
    <Alert className="w-full">
    
      <LinkNone2Icon className="h-4 w-4" />
      <AlertTitle>      <div style={{color: 'green'}}>Connected</div></AlertTitle>
      {/* <AlertDescription>
      <div style={{color: 'green'}}>Connected</div>
      </AlertDescription> */}

    </Alert>
  )
}
