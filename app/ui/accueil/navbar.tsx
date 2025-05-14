import { ericaOne } from "../fonts"
import Link from "next/link"

export default function Navbar() {
    return (
        <div className="flex items-center p-5 text-white border-b-2 border-[#404040] bg-[#070707] fixed w-full px-40 justify-between ">
            <div>
                <h1 className={`text-3xl ${ericaOne.className}`}>HarmonIA</h1>
            </div>
            <div className="ml-5 text-base">
                <ul className="flex">
                    <li className="mr-10">Accueil</li>
                    <li className="mr-10">À propos</li>
                    <li className="mr-10">Contact</li>
                </ul>
            </div>
        </div>
    )
}