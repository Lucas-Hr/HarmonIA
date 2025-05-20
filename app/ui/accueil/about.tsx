import React from 'react'
import Image from 'next/image'
import { motion } from 'framer-motion'
const About = () => {
  return (
    <div id="about" className='px-40 py-40 '>
        <div className='flex justify-between mt-16 '>
            <motion.div 
             initial={{opacity : 0, y : 50}}
             whileInView={{opacity : 1, y : 0}}
             transition={{
                duration : 1,
                delay : 0.25
             }}
             viewport={{once:true}}
             >
                <Image src="/music.png" width={2000} height={100} alt='guitarist' className='h-96'/>
            </motion.div>
            
            <div className=''>
                <motion.h1 
                initial={{opacity : 0, y : 50}}
                whileInView={{opacity : 1, y : 0}}
                transition={{
                   duration : 1,
                   delay : 0.5
                }}
                viewport={{ once: true }}
                className='text-3xl font-bold'>A propos</motion.h1>
                <motion.p 
                initial={{opacity : 0, y : 50}}
                whileInView={{opacity : 1, y : 0}}
                transition={{
                   duration : 1,
                   delay : 0.75
                }}
                viewport={{ once: true }}
                className='text-[#B6B6B6] text-md font-lighter lien text-base/8 mt-6'>
                    <span className='text-white font-semibold'>Découvrez HarmonIA – Votre assistant musical intelligent !</span><br />
                    HarmonIA est une intelligence artificielle conçue pour simplifier la vie des musiciens, en particulier les amateurs. Vous avez une partition difficile à déchiffrer ? HarmonIA la transforme en musique, pour que vous puissiez l’écouter et mieux la comprendre.
                    Vous avez une chanson en tête mais vous ne connaissez pas les notes ? HarmonIA la convertit en partition claire et précise, adaptée à votre instrument.
                </motion.p>
                <motion.p
                initial={{opacity : 0, y : 50}}
                whileInView={{opacity : 1, y : 0}}
                transition={{
                   duration : 1,
                   delay : 1
                }}
                viewport={{ once: true }}
                className='text-[#B6B6B6] text-md     font-lighter lien text-base/8 mt-6'>
                    <span className='text-white font-semibold'>Écoutez, apprenez, jouez.</span> <br />
                    Avec HarmonIA, jouer de la musique devient plus simple, plus intuitif et plus agréable. Que vous appreniez un nouvel instrument ou que vous souhaitiez rejouer vos morceaux préférés, HarmonIA est là pour vous guider, note après note.
                </motion.p>
            </div>
        </div>
    </div>

  )
}

export default About