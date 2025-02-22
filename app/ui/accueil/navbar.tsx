import { ericaOne } from "../fonts"

export default function Navbar() {
    return (
        <div className="flex items-center p-5 text-white border-b-2 border-[#404040] bg-[#070707] fixed w-full px-40 ">
            <div>
                <h1 className={`text-3xl ${ericaOne.className}`}>HarmonIA</h1>
            </div>
            <div className="ml-5 text-base">
                <ul className="flex">
                    <li className="mr-6">Accueil</li>
                    <li className="mr-6">À propos</li>
                    <li className="mr-6">Contact</li>
                </ul>
            </div>
        </div>
    )
}