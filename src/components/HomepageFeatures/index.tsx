import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';
type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Team',
    Svg: require('@site/static/img/team2.svg').default,
    description: (
      <>
        BrickBar is made up of students who study AI and keep a learning heart in their daily brick-lifting lives<code>Team</code>.
      </>
    ),
  },
  {
    title: 'Focus on What Matters',
    Svg: require('@site/static/img/what2.svg').default,
    description: (
      <>
        Bricklayers share study notes they write <code>Note</code>, papers they read<code>Paper</code>, 
        novel knowledge they see<code>News</code>, and messy insights<code>Hodgepodge</code> in BrickBar.
      </>
    ),
  },
  {
    title: 'Powered by Passion',
    Svg: require('@site/static/img/why.svg').default,
    description: (
      <>
        BrickBar is maintained by bricklayers in their spare time, 
        driven entirely by love ❤️ and passion to document the learning process.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
