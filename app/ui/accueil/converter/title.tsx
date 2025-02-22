import Image from "next/image"

export default function Title({text1, text2} : {text1: string, text2: string}) {
    return (
        <div className="flex justify-between items-center">
            <h1 className="font-extralight text-5xl">{text1}</h1>
            <Image
                src="/arrow.png"
                alt="arrow"
                width={130}
                height={130}
                className="transform rotate-[-90deg]"
            />
            <h1 className="font-extralight text-5xl">{text2}</h1>
        </div>
    )
}