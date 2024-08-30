import { AcademicPage } from "@/components/academic_page_v2/acad-page_2";
import { HeaderComponent } from "@/components/header-component";
import { HomePage } from "@/components/home-page";
import { Button } from "@/components/ui/button";
import Link from "next/link"
import { SheetTrigger, SheetContent, Sheet } from "@/components/ui/sheet"
import { Label } from "@/components/ui/label"
import { DropdownMenuTrigger, DropdownMenuRadioItem, DropdownMenuRadioGroup, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { TableHead, TableRow, TableHeader, TableCell, TableBody, Table } from "@/components/ui/table"
import { AvatarImage, AvatarFallback, Avatar } from "@/components/ui/avatar"
import { MenuIcon, TrophyIcon } from "@/components/leader-board";


export default function Home() {
  return (
 <div>
    <header className="flex h-16 items-center justify-between px-4 md:px-6 border-b">
        <Link className="flex items-center gap-2" href="/">
          <TrophyIcon className="h-6 w-6" />
          <span className="font-bold">omni_epic</span>
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
              <Link href="#">Contact</Link>
            </div>
          </SheetContent>
        </Sheet>
      </header>
  <section className="bg-gray-100 dark:bg-gray-800 py-12 md:py-20">
        {/* <div className="container">
          <div className="max-w-3xl mx-auto space-y-6 text-center">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
             OMNI EPIC
              <sup>1</sup>
            </h1>
            <div className="text-gray-500 dark:text-gray-400 space-x-4">
              <span>Name 1</span>
              <span>Name 2</span>
            </div>
            <p className="text-gray-500 dark:text-gray-400">
              Published in NeurIPS 2024
            </p>
          </div>
        </div> */}
        <HeaderComponent></HeaderComponent>
      </section>
  <HomePage></HomePage>
  <div className="p-32"></div>
  <AcademicPage></AcademicPage>
 </div>
 
    );
}
