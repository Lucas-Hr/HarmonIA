import Card from "./card"
import Link from "next/link"
import { motion } from "framer-motion"

export default function Choix() {
    return (
        <div className="px-40 h-screen py-40 bg-[black]">
            <motion.h1
            initial={{opacity : 0, y : 50}}
            whileInView={{opacity : 1, y : 0}}
            transition={{
               duration : 1,
               delay : 0.25
            }}
            className="text-white text-5xl text-center font-lighter">Que voulez-vous faire?</motion.h1>
            <motion.div 
                initial={{opacity : 0, y : 50}}
                whileInView={{opacity : 1, y : 0}}
                transition={{
                duration : 1,
                delay : 0.5
                }}
            className="flex justify-evenly items-center mt-20">
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
            </motion.div>
        </div>
    )
}