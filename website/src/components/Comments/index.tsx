import React from 'react';
import Giscus from '@giscus/react';
import {useColorMode} from '@docusaurus/theme-common';

export default function Comments(): JSX.Element {
  const {colorMode} = useColorMode();

  return (
    <div style={{marginTop: '2rem', paddingTop: '2rem', borderTop: '1px solid var(--ifm-color-emphasis-300)'}}>
      <Giscus
        repo="blakechasteen/hello-world"
        repoId=""
        category="Documentation"
        categoryId=""
        mapping="pathname"
        strict="0"
        reactionsEnabled="1"
        emitMetadata="0"
        inputPosition="top"
        theme={colorMode === 'dark' ? 'dark' : 'light'}
        lang="en"
        loading="lazy"
      />
    </div>
  );
}
