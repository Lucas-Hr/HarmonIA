import Image from "next/image"
import { motion } from "framer-motion"
export default function Title({text1, text2} : {text1: string, text2: string}) {
    return (
        <motion.div 
        initial={{opacity : 0, y : 50}}
        animate={{opacity : 1, y : 0}}
        transition={{
            duration : 1
        }}
        className="flex justify-between items-center">
            <h1 className="font-extralight text-5xl">{text1}</h1>
            <Image
                src="/arrow.png"
                alt="arrow"
                width={130}
                height={130}
                className="transform rotate-[-90deg]"
            />
            <h1 className="font-extralight text-5xl">{text2}</h1>
        </motion.div>
    )
}