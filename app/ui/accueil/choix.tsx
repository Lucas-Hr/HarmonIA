import Card from "./card"
import Link from "next/link"
import { motion } from "framer-motion"

export default function Choix() {
    return (
        <div className="px-40 h-screen py-40 bg-[black]">
            <h1 className="text-white text-5xl text-center font-lighter">Que voulez-vous faire?</h1>
            <div className="flex justify-evenly items-center mt-20">
                <Link href={{
                    pathname : './accueil/converter',
                    query : {textOne : "Musique", textTwo : "Partition"}
                }} className="cursor-pointer">
                    <Card text1="Musique" text2="Partition"/>
                </Link>
                <Link href={{
                    pathname : './accueil/converter',
                    query : {textOne : "Partition", textTwo : "Musique"}
                }} className="cursor-pointer">
                    <Card text1="Partition" text2="Musique"/>
                </Link>
            </div>
        </div>
    )
}