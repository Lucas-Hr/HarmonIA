import { ericaOne } from "../fonts"
import Link from "next/link"
import Image from "next/image"

export default function Navbar() {
    const links = [
        {
        name: "Accueil",
        href : "/accueil"
        },
        {
        name: "A propos",
        href : "/accueil#about"
        },
        {
        name: "Testimonials",
        href : "/accueil#testimonials"
        },
    ]
    return (
        <div className="flex items-center p-5 text-white border-b-2 border-[#404040] bg-[#070707] fixed w-full px-40 justify-between z-4 relative">
            <div className="flex ">
                {/* <Image src='/ispm.png' width={50} height={20} alt="ispm" /> */}
                <h1 className={`text-3xl ${ericaOne.className}`}>HarmonIA</h1>
            </div>
            <div className="ml-5 text-base flex">
                    {links.map((link) => {
                        return (
                            <Link
                            key={link.name}
                            href={link.href}
                            className="ms-10 hover:font-bold"
                            >
                            <p className="">{link.name}</p>
                            </Link>
                        );
                })}
            </div>
        </div>
    )
}