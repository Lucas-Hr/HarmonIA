import React from 'react'
import Image from 'next/image'
const About = () => {
  return (
    <div className='px-40 py-20 h-screen'>
        <div className='flex justify-between mt-16 '>
            <Image src="/music.png" width={800} height={100} alt='guitarist' className='h-96'/>
            <div className=''>
                <h1 className='text-3xl font-bold'>A propos</h1>
                <p className='text-[#B6B6B6] text-md font-lighter lien text-base/8 mt-6'>
                    <span className='text-white font-semibold'>Découvrez HarmonIA – Votre assistant musical intelligent !</span><br />
                    HarmonIA est une intelligence artificielle conçue pour simplifier la vie des musiciens, en particulier les amateurs. Vous avez une partition difficile à déchiffrer ? HarmonIA la transforme en musique, pour que vous puissiez l’écouter et mieux la comprendre.
                    Vous avez une chanson en tête mais vous ne connaissez pas les notes ? HarmonIA la convertit en partition claire et précise, adaptée à votre instrument.
                </p>
                <p className='text-[#B6B6B6] text-md font-lighter lien text-base/8 mt-6'>
                    <span className='text-white font-semibold'>Écoutez, apprenez, jouez.</span> <br />
                    Avec HarmonIA, jouer de la musique devient plus simple, plus intuitif et plus agréable. Que vous appreniez un nouvel instrument ou que vous souhaitiez rejouer vos morceaux préférés, HarmonIA est là pour vous guider, note après note.
                </p>
            </div>
        </div>
    </div>

  )
}

export default About