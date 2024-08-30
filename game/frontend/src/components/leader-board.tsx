
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { SheetTrigger, SheetContent, Sheet } from "@/components/ui/sheet"
import { Label } from "@/components/ui/label"
import { DropdownMenuTrigger, DropdownMenuRadioItem, DropdownMenuRadioGroup, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { TableHead, TableRow, TableHeader, TableCell, TableBody, Table } from "@/components/ui/table"
import { AvatarImage, AvatarFallback, Avatar } from "@/components/ui/avatar"

export function LeaderBoard() {
  return (
    <>
      <header className="flex h-16 items-center justify-between px-4 md:px-6 border-b">
        <Link className="flex items-center gap-2" href="/">
          <TrophyIcon className="h-6 w-6" />
          <span className="font-bold">OMNI EPIC Leaderboard</span>
        </Link>
        <Sheet>
          <SheetTrigger asChild>
            <Button size="icon" variant="outline">
              <MenuIcon className="h-6 w-6" />
              <span className="sr-only">Toggle navigation menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="right">
            <div className="grid gap-4 p-4">
              <Link href="/">Home</Link>
              <Link href="/leaderboard">Leaderboard</Link>
              <Link href="/about">About</Link>
              <Link href="/contact">Contact</Link>
            </div>
          </SheetContent>
        </Sheet>
      </header>
      <main className="container px-4 md:px-6 py-8">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Leaderboard</h1>
            <div className="flex items-center gap-2">
              <Label className="text-sm" htmlFor="sort">
                Sort by:
              </Label>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="sm" variant="outline">
                    <ArrowUpDownIcon className="h-4 w-4 mr-2" />
                    Score
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-40">
                  <DropdownMenuRadioGroup value="score">
                    <DropdownMenuRadioItem value="score">Score</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="rank">Rank</DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="name">Name</DropdownMenuRadioItem>
                  </DropdownMenuRadioGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[80px]">Rank</TableHead>
                <TableHead>Player</TableHead>
                <TableHead className="text-right">Score</TableHead>
                <TableHead className="text-right">Wins</TableHead>
                <TableHead className="text-right">Losses</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">1</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Avatar>
                      <AvatarImage alt="Player 1" src="/placeholder-avatar.jpg" />
                      <AvatarFallback>P1</AvatarFallback>
                    </Avatar>
                    <span>Player 1</span>
                  </div>
                </TableCell>
                <TableCell className="text-right">12,345</TableCell>
                <TableCell className="text-right">100</TableCell>
                <TableCell className="text-right">25</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">2</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Avatar>
                      <AvatarImage alt="Player 2" src="/placeholder-avatar.jpg" />
                      <AvatarFallback>P2</AvatarFallback>
                    </Avatar>
                    <span>Player 2</span>
                  </div>
                </TableCell>
                <TableCell className="text-right">11,987</TableCell>
                <TableCell className="text-right">95</TableCell>
                <TableCell className="text-right">30</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">3</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Avatar>
                      <AvatarImage alt="Player 3" src="/placeholder-avatar.jpg" />
                      <AvatarFallback>P3</AvatarFallback>
                    </Avatar>
                    <span>Player 3</span>
                  </div>
                </TableCell>
                <TableCell className="text-right">10,654</TableCell>
                <TableCell className="text-right">90</TableCell>
                <TableCell className="text-right">35</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">4</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Avatar>
                      <AvatarImage alt="Player 4" src="/placeholder-avatar.jpg" />
                      <AvatarFallback>P4</AvatarFallback>
                    </Avatar>
                    <span>Player 4</span>
                  </div>
                </TableCell>
                <TableCell className="text-right">9,876</TableCell>
                <TableCell className="text-right">85</TableCell>
                <TableCell className="text-right">40</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">5</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Avatar>
                      <AvatarImage alt="Player 5" src="/placeholder-avatar.jpg" />
                      <AvatarFallback>P5</AvatarFallback>
                    </Avatar>
                    <span>Player 5</span>
                  </div>
                </TableCell>
                <TableCell className="text-right">8,765</TableCell>
                <TableCell className="text-right">80</TableCell>
                <TableCell className="text-right">45</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </main>
    </>
  )
}

function ArrowUpDownIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m21 16-4 4-4-4" />
      <path d="M17 20V4" />
      <path d="m3 8 4-4 4 4" />
      <path d="M7 4v16" />
    </svg>
  )
}


export function MenuIcon(props:any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="4" x2="20" y1="12" y2="12" />
      <line x1="4" x2="20" y1="6" y2="6" />
      <line x1="4" x2="20" y1="18" y2="18" />
    </svg>
  )
}


export function TrophyIcon(props:any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
      <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
      <path d="M4 22h16" />
      <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" />
      <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" />
      <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
    </svg>
  )
}
