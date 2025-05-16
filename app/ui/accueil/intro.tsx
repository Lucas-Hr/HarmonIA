'use client'
import { motion } from "framer-motion"


const Intro = () => {
    return(
        <div className="px-40 h-screen bg-piano bg-cover bg-fixed pt-60">
            <motion.h1 
            className="text-white font-bold text-center text-7xl"
            initial={{opacity : 0, y : 50}}
            animate={{opacity : 1, y : 0}}
            transition={{
                duration : 1
            }}
            >Lisez vos partitions, Jouez votre musique !</motion.h1>
            <motion.p 
             initial={{opacity : 0, y : 50}}
             animate={{opacity : 1, y : 0}}
             transition={{
                 duration : 1,
                 delay : 0.5
             }}
            className="text-[#B6B6B6] text-xl text-center mt-8 font-extralighter">Notre plateforme vous permet de convertir vos partitions en fichiers audio, et d'exporter vos fichiers audio en partitions musicales.
                Profitez d'une expérience unique pour explorer la musique sous toutes ses formes.</motion.p>
            <motion.div
            initial={{opacity : 0, y : 50}}
            animate={{opacity : 1, y : 0}}
            transition={{
                duration : 1,
                delay : 1
            }}
            className="flex justify-center mt-8">
                <button className="bg-white text-black px-5 py-2 rounded-lg ">Commencer</button>
            </motion.div>   
        </div>
    )
}

export default Intro;